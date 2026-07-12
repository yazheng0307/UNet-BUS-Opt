import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def group_count(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class GNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class GNUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.fusion = GNResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.projection(x)
        return self.fusion(torch.cat([skip, x], dim=1))


class ConvNeXtTinyUNet(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny encoder with a GroupNorm U-Net decoder."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=pretrained,
            features_only=True,
            pretrained_cfg_overlay={"hf_hub_id": None},
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.input_skip = GNResidualBlock(3, 32)
        self.decoder3 = GNUpBlock(768, 384, 384)
        self.decoder2 = GNUpBlock(384, 192, 192)
        self.decoder1 = GNUpBlock(192, 96, 96)
        self.half_decoder = GNResidualBlock(96, 64)
        self.full_decoder = GNUpBlock(64, 32, 32)
        self.segmentation_head = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        full_resolution = self.input_skip(image)
        features = self.encoder((image - self.image_mean) / self.image_std)
        x = self.decoder3(features[3], features[2])
        x = self.decoder2(x, features[1])
        x = self.decoder1(x, features[0])
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.half_decoder(x)
        x = self.full_decoder(x, full_resolution)
        return self.segmentation_head(x)
