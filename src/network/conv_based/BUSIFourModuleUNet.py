import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.BUSIOptUNet import BUSIOptUNet, ConvNormAct, ResidualConvBlock


class BoundaryGuidedUpBlock(nn.Module):
    """Decoder block whose coarse semantic edge gates the encoder skip."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.projection = ConvNormAct(in_channels, out_channels, 3)
        self.coarse_boundary = nn.Conv2d(out_channels, 1, 1)
        hidden = max(skip_channels // 4, 8)
        self.skip_gate = nn.Sequential(
            nn.Conv2d(skip_channels + 1, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, skip_channels, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.skip_gate[-2].weight)
        nn.init.zeros_(self.skip_gate[-2].bias)
        self.fusion = ResidualConvBlock(out_channels + skip_channels, out_channels)
        self.refined_boundary = nn.Conv2d(out_channels, 1, 1)
        self.distance = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.projection(x)
        coarse_boundary = self.coarse_boundary(x)
        gate = self.skip_gate(torch.cat([skip, torch.sigmoid(coarse_boundary)], dim=1))
        gated_skip = skip * (0.5 + gate)
        fused = self.fusion(torch.cat([gated_skip, x], dim=1))
        boundary = coarse_boundary + self.refined_boundary(fused)
        distance = torch.tanh(self.distance(fused))
        return fused, boundary, distance


class BoundaryDistanceCooperativeHead(nn.Module):
    """Boundary-distance cooperative logit correction at full resolution."""

    def __init__(self, channels):
        super().__init__()
        self.correction = nn.Sequential(
            ConvNormAct(channels + 2, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )
        self.scale = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, feature, coarse_logit, boundary_logit, distance):
        if distance is None:
            distance = torch.zeros_like(boundary_logit)
        geometry = torch.cat([feature, torch.sigmoid(boundary_logit), distance], dim=1)
        correction = self.correction(geometry)
        return coarse_logit + torch.sigmoid(self.scale) * correction


class UncertaintyDrivenDualErrorRefinement(nn.Module):
    """Predict separate false-negative additions and false-positive removals."""

    def __init__(self, channels, version=1):
        super().__init__()
        self.version = version
        context_channels = max(channels, 32)
        self.context = nn.Sequential(
            ConvNormAct(channels + 5, context_channels, 3),
            ConvNormAct(context_channels, context_channels, 3, dilation=2),
        )
        self.false_negative_head = nn.Conv2d(context_channels, 1, 1)
        self.false_positive_head = nn.Conv2d(context_channels, 1, 1)
        self.correction_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.false_negative_head.weight)
        nn.init.zeros_(self.false_negative_head.bias)
        nn.init.zeros_(self.false_positive_head.weight)
        nn.init.zeros_(self.false_positive_head.bias)

    def forward(self, feature, coarse_logit, auxiliary_logits, boundary_logit=None, distance=None):
        probability = torch.sigmoid(coarse_logit)
        uncertainty_probability = probability.detach() if self.version >= 2 else probability
        entropy = -uncertainty_probability * torch.log(uncertainty_probability.clamp_min(1e-6))
        entropy -= (1.0 - uncertainty_probability) * torch.log(
            (1.0 - uncertainty_probability).clamp_min(1e-6)
        )
        entropy = entropy / math.log(2.0)

        probabilities = [uncertainty_probability]
        for auxiliary in auxiliary_logits:
            auxiliary = F.interpolate(auxiliary, size=coarse_logit.shape[-2:], mode="bilinear", align_corners=False)
            auxiliary_probability = torch.sigmoid(auxiliary)
            probabilities.append(auxiliary_probability.detach() if self.version >= 2 else auxiliary_probability)
        disagreement = torch.stack(probabilities, dim=0).std(dim=0, unbiased=False)
        uncertainty = (0.7 * entropy + 0.3 * disagreement).clamp(0.0, 1.0)
        if boundary_logit is None:
            boundary_probability = torch.zeros_like(probability)
        else:
            boundary_probability = torch.sigmoid(boundary_logit)
        if distance is None:
            distance = torch.zeros_like(probability)

        context = self.context(
            torch.cat(
                [feature, probability, entropy, disagreement, boundary_probability, distance],
                dim=1,
            )
        )
        false_negative = self.false_negative_head(context)
        false_positive = self.false_positive_head(context)
        if self.version >= 2:
            residual = 0.5 * uncertainty * (
                torch.tanh(false_negative) - torch.tanh(false_positive)
            )
        else:
            residual = uncertainty * (torch.sigmoid(false_negative) - torch.sigmoid(false_positive))
        scale = 2.0 * torch.sigmoid(self.correction_scale)
        corrected = coarse_logit + scale * residual
        return corrected, uncertainty, false_negative, false_positive


class BUSIFourModuleUNet(nn.Module):
    def __init__(
        self,
        output_channels=1,
        base_channels=32,
        use_sfs=False,
        use_tricar=False,
        use_router=None,
        use_graph=None,
        sfs_version=2,
        use_bd_code=False,
        use_distance=True,
        use_uder=False,
        uder_version=1,
    ):
        super().__init__()
        if output_channels != 1:
            raise ValueError("The four-module BUSI model currently supports binary segmentation only")
        self.use_bd_code = use_bd_code
        self.use_distance = use_distance
        self.use_uder = use_uder
        self.backbone = BUSIOptUNet(
            output_channels=output_channels,
            base_channels=base_channels,
            use_sfs=use_sfs,
            use_tricar=use_tricar,
            sfs_version=sfs_version,
            use_router=use_router,
            use_graph=use_graph,
        )
        channels = [base_channels * (2 ** index) for index in range(5)]
        decoder_channels = [channels[3], channels[2], channels[1], channels[0]]
        if use_bd_code:
            self.backbone.decoders = nn.ModuleList()
            self.boundary_decoders = nn.ModuleList(
                [
                    BoundaryGuidedUpBlock(channels[4], channels[3], channels[3]),
                    BoundaryGuidedUpBlock(channels[3], channels[2], channels[2]),
                    BoundaryGuidedUpBlock(channels[2], channels[1], channels[1]),
                    BoundaryGuidedUpBlock(channels[1], channels[0], channels[0]),
                ]
            )
            self.geometry_head = BoundaryDistanceCooperativeHead(channels[0])
        else:
            self.boundary_decoders = None
            self.geometry_head = None

        self.auxiliary_heads = nn.ModuleList([nn.Conv2d(channel, 1, 1) for channel in decoder_channels[:-1]])
        self.uder = (
            UncertaintyDrivenDualErrorRefinement(channels[0], version=uder_version)
            if use_uder else None
        )

    def _decode(self, features):
        if self.boundary_decoders is None:
            decoded, decoder_features = self.backbone.decode(features)
            return decoded, decoder_features, [], []
        x = features[-1]
        decoder_features = []
        boundaries = []
        distances = []
        for decoder, skip in zip(self.boundary_decoders, reversed(features[:-1])):
            x, boundary, distance = decoder(x, skip)
            decoder_features.append(x)
            boundaries.append(boundary)
            distances.append(distance)
        if not self.use_distance:
            distances = []
        return x, decoder_features, boundaries, distances

    def forward(self, image):
        features = self.backbone.encode(image)
        decoded, decoder_features, boundaries, distances = self._decode(features)
        coarse_logit = self.backbone.segmentation_head(decoded)
        auxiliary_logits = [
            head(feature) for head, feature in zip(self.auxiliary_heads, decoder_features[:-1])
        ]

        segmentation = coarse_logit
        boundary_logit = boundaries[-1] if boundaries else None
        distance = distances[-1] if distances else None
        if self.geometry_head is not None:
            segmentation = self.geometry_head(decoded, segmentation, boundary_logit, distance)

        uncertainty = None
        false_negative = None
        false_positive = None
        if self.uder is not None:
            segmentation, uncertainty, false_negative, false_positive = self.uder(
                decoded,
                segmentation,
                auxiliary_logits,
                boundary_logit,
                distance,
            )

        return {
            "segmentation": segmentation,
            "coarse_segmentation": coarse_logit,
            "auxiliary_segmentations": auxiliary_logits,
            "boundary_logits": boundaries,
            "distance_maps": distances,
            "uncertainty": uncertainty,
            "false_negative_logits": false_negative,
            "false_positive_logits": false_positive,
            "uder_version": self.uder.version if self.uder is not None else 0,
        }


def build_busi_four_module_unet(variant="B3", base_channels=32):
    variants = {
        "B0": (False, False, False, 2, False, False, False, 1),
        "B1": (False, False, False, 2, True, False, False, 1),
        "B1D": (False, False, False, 2, True, True, False, 1),
        "B2": (False, False, False, 2, False, False, True, 1),
        "B2V2": (False, False, False, 2, False, False, True, 2),
        "B3": (False, False, False, 2, True, True, True, 1),
        "B3V2": (False, False, False, 2, True, True, True, 2),
        "FG": (False, False, True, 2, True, True, True, 1),
        "FGV2": (False, False, True, 2, True, True, True, 2),
        "FR": (False, True, False, 2, True, True, True, 1),
        "F": (False, True, True, 2, True, True, True, 1),
        "FLEGACY": (True, True, True, 1, True, True, True, 1),
    }
    if variant not in variants:
        raise ValueError("Unknown four-module variant: {}".format(variant))
    use_sfs, use_router, use_graph, sfs_version, use_bd_code, use_distance, use_uder, uder_version = variants[variant]
    return BUSIFourModuleUNet(
        base_channels=base_channels,
        use_sfs=use_sfs,
        use_tricar=use_router and use_graph,
        use_router=use_router,
        use_graph=use_graph,
        sfs_version=sfs_version,
        use_bd_code=use_bd_code,
        use_distance=use_distance,
        use_uder=use_uder,
        uder_version=uder_version,
    )
