import torch
import torch.nn as nn

from src.network.conv_based.BUSIFourModuleUNet import (
    BoundaryDistanceCooperativeHead,
    UncertaintyDrivenDualErrorRefinement,
)
from src.network.conv_based.BUSIOptUNet import ChannelGraphReasoning
from src.network.conv_based.ResNet50UNet import DEFAULT_DEEPLAB_CHECKPOINT, ResNet50UNet


class ResNet50FourModuleUNet(nn.Module):
    """Non-destructive geometry/uncertainty refinement on the strong U-Net baseline."""

    def __init__(
        self,
        encoder_checkpoint=DEFAULT_DEEPLAB_CHECKPOINT,
        use_graph=True,
        use_geometry=True,
        use_uder=True,
        use_deep_supervision=None,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_geometry = use_geometry
        self.use_uder = use_uder
        self.use_deep_supervision = (
            use_uder if use_deep_supervision is None else use_deep_supervision
        )
        self.backbone = ResNet50UNet(checkpoint_path=encoder_checkpoint)
        decoder_channels = (256, 128, 64, 32, 32)
        self.graph_bridge = ChannelGraphReasoning(512)
        self.auxiliary_heads = nn.ModuleList(
            [nn.Conv2d(channel, 1, 1) for channel in decoder_channels[:-1]]
        )
        self.boundary_heads = nn.ModuleList(
            [nn.Conv2d(channel, 1, 1) for channel in decoder_channels]
        )
        self.distance_heads = nn.ModuleList(
            [nn.Conv2d(channel, 1, 1) for channel in decoder_channels]
        )
        self.geometry_head = BoundaryDistanceCooperativeHead(decoder_channels[-1])
        self.uder = UncertaintyDrivenDualErrorRefinement(decoder_channels[-1], version=1)

    def load_segmentation_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = self.backbone.load_state_dict(state, strict=True)
        return len(state), missing, unexpected

    def forward(self, image):
        full_resolution, features = self.backbone.encode(image)
        if self.use_graph:
            features[-1] = self.graph_bridge(features[-1])
        decoded, decoder_features = self.backbone.decode(full_resolution, features)
        coarse = self.backbone.segmentation_head(decoded)
        auxiliary = (
            [head(feature) for head, feature in zip(self.auxiliary_heads, decoder_features[:-1])]
            if self.use_deep_supervision else []
        )
        boundaries = (
            [head(feature) for head, feature in zip(self.boundary_heads, decoder_features)]
            if self.use_geometry else []
        )
        distances = (
            [torch.tanh(head(feature)) for head, feature in zip(self.distance_heads, decoder_features)]
            if self.use_geometry else []
        )
        segmentation = (
            self.geometry_head(decoded, coarse, boundaries[-1], distances[-1])
            if self.use_geometry else coarse
        )
        if self.use_uder:
            segmentation, uncertainty, false_negative, false_positive = self.uder(
                decoded,
                segmentation,
                auxiliary,
                boundaries[-1] if boundaries else None,
                distances[-1] if distances else None,
            )
        else:
            uncertainty = None
            false_negative = None
            false_positive = None
        return {
            "segmentation": segmentation,
            "coarse_segmentation": coarse,
            "auxiliary_segmentations": auxiliary,
            "boundary_logits": boundaries,
            "distance_maps": distances,
            "uncertainty": uncertainty,
            "false_negative_logits": false_negative,
            "false_positive_logits": false_positive,
            "uder_version": 1,
        }
