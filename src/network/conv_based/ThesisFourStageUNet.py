import torch
import torch.nn as nn

from src.network.conv_based.BUSIFourModuleUNet import (
    BoundaryDistanceCooperativeHead,
    UncertaintyDrivenDualErrorRefinement,
)
from src.network.conv_based.BUSIOptUNet import (
    BUSIOptUNet,
    ChannelGraphReasoning,
    ConvNormAct,
    DepthwiseExpert,
)


class StableTriChallengeAdapter(nn.Module):
    """Module A: identity-initialized routing for noise, small lesions, and boundaries."""

    def __init__(self, channels):
        super().__init__()
        self.noise_expert = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            DepthwiseExpert(channels, kernel_size=3),
        )
        self.small_lesion_expert = DepthwiseExpert(channels, kernel_size=3)
        self.boundary_expert = DepthwiseExpert(channels, kernel_size=3, dilation=3)
        hidden = max(channels // 4, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        self.fusion = nn.Sequential(
            ConvNormAct(channels * 2, channels, 1),
            nn.Conv2d(channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, feature):
        weights = torch.softmax(self.router(feature), dim=1)
        experts = (
            self.noise_expert(feature),
            self.small_lesion_expert(feature),
            self.boundary_expert(feature),
        )
        mixed = sum(
            expert * weights[:, index:index + 1]
            for index, expert in enumerate(experts)
        )
        correction = self.fusion(torch.cat([feature, mixed], dim=1))
        return feature + torch.sigmoid(self.scale) * correction


class ThesisFourStageUNet(nn.Module):
    """One parameter-compatible model for the complete U/UA/UB/UAB/UABC/UABCD study."""

    VARIANTS = {
        "U": (False, False, False, False),
        "UA": (True, False, False, False),
        "UB": (False, True, False, False),
        "UAB": (True, True, False, False),
        "UABC": (True, True, True, False),
        "UABCD": (True, True, True, True),
    }

    def __init__(self, variant="U", base_channels=32):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError("Unknown thesis variant: {}".format(variant))
        self.variant = variant
        self.use_a, self.use_b, self.use_c, self.use_d = self.VARIANTS[variant]
        self.backbone = BUSIOptUNet(
            base_channels=base_channels,
            use_sfs=False,
            use_tricar=False,
            use_router=False,
            use_graph=False,
        )
        channels = [base_channels * (2 ** index) for index in range(5)]
        decoder_channels = [channels[3], channels[2], channels[1], channels[0]]

        self.module_a = nn.ModuleList(
            [StableTriChallengeAdapter(channel) for channel in channels]
        )
        self.module_b = ChannelGraphReasoning(channels[-1])
        self.boundary_head = nn.Conv2d(channels[0], 1, 1)
        self.distance_head = nn.Conv2d(channels[0], 1, 1)
        self.module_c = BoundaryDistanceCooperativeHead(channels[0])
        self.auxiliary_heads = nn.ModuleList(
            [nn.Conv2d(channel, 1, 1) for channel in decoder_channels[:-1]]
        )
        self.module_d = UncertaintyDrivenDualErrorRefinement(channels[0], version=1)

    def load_progressive_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        current = self.state_dict()
        if any(key.startswith("backbone.") for key in state):
            compatible = {
                key: value for key, value in state.items()
                if key in current and current[key].shape == value.shape
            }
        else:
            compatible = {
                "backbone." + key: value for key, value in state.items()
                if "backbone." + key in current
                and current["backbone." + key].shape == value.shape
            }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return len(compatible), missing, unexpected

    def encode(self, image):
        features = []
        x = image
        for index, encoder in enumerate(self.backbone.encoders):
            if index:
                x = self.backbone.pool(x)
            x = encoder(x)
            if self.use_a:
                x = self.module_a[index](x)
            features.append(x)
        if self.use_b:
            features[-1] = self.module_b(features[-1])
        return features

    def forward(self, image):
        features = self.encode(image)
        decoded, decoder_features = self.backbone.decode(features)
        coarse = self.backbone.segmentation_head(decoded)

        boundaries = []
        distances = []
        segmentation = coarse
        if self.use_c:
            boundary = self.boundary_head(decoded)
            distance = torch.tanh(self.distance_head(decoded))
            boundaries = [boundary]
            distances = [distance]
            segmentation = self.module_c(decoded, coarse, boundary, distance)

        auxiliary = []
        uncertainty = None
        false_negative = None
        false_positive = None
        if self.use_d:
            auxiliary = [
                head(feature)
                for head, feature in zip(self.auxiliary_heads, decoder_features[:-1])
            ]
            segmentation, uncertainty, false_negative, false_positive = self.module_d(
                decoded,
                segmentation,
                auxiliary,
                boundaries[-1] if boundaries else None,
                distances[-1] if distances else None,
            )

        return {
            "segmentation": segmentation,
            "coarse_segmentation": coarse,
            "auxiliary_segmentations": auxiliary,
            "boundary_logits": boundaries,
            "distance_maps": distances,
            "uncertainty": uncertainty,
            "false_negative_logits": false_negative,
            "false_positive_logits": false_positive,
            "uder_version": 1 if self.use_d else 0,
        }
