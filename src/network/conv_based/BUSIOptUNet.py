import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class HaarHighFrequency(nn.Module):
    def __init__(self):
        super().__init__()
        scale = 0.5
        filters = torch.tensor(
            [
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ) * scale
        self.register_buffer("filters", filters[:, None], persistent=False)

    def forward(self, image):
        channels = image.shape[1]
        filters = self.filters.repeat(channels, 1, 1, 1)
        return F.conv2d(image, filters, stride=2, groups=channels).abs()


class HaarFrequencyDecomposition(nn.Module):
    """Luminance Haar bands used by the speckle-shrinkage calibration path."""

    def __init__(self):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ) * 0.5
        self.register_buffer("filters", filters[:, None], persistent=False)

    def forward(self, image):
        luminance = image.mean(dim=1, keepdim=True)
        return F.conv2d(luminance, self.filters, stride=2)


class SFSCalibration(nn.Module):
    """Speckle-robust frequency-spatial calibration (SFS-Cal)."""

    def __init__(self, channels, frequency_channels=9):
        super().__init__()
        self.spatial_local = ConvNormAct(channels, channels, 3, groups=channels)
        self.spatial_context = ConvNormAct(channels, channels, 3, dilation=2, groups=channels)
        self.spatial_fuse = ConvNormAct(channels * 2, channels, 1)
        self.frequency_projection = nn.Sequential(
            ConvNormAct(frequency_channels, channels, 3),
            ConvNormAct(channels, channels, 3, groups=channels),
        )
        hidden = max(channels // 4, 8)
        self.calibration_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.output = ResidualConvBlock(channels, channels)

    def forward(self, spatial, high_frequency):
        local = self.spatial_local(spatial)
        context = self.spatial_context(spatial)
        spatial_feature = self.spatial_fuse(torch.cat([local, context], dim=1))
        frequency = F.interpolate(
            high_frequency,
            size=spatial.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        frequency_feature = self.frequency_projection(frequency)
        gate = self.calibration_gate(torch.cat([spatial_feature, frequency_feature], dim=1))
        return self.output(spatial + spatial_feature + gate * frequency_feature)


class SpeckleShrinkageSFSCalibration(nn.Module):
    """Residual SFS-Cal v2 with adaptive wavelet shrinkage and spatial gating."""

    def __init__(self, channels):
        super().__init__()
        self.threshold_logits = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.spatial_local = ConvNormAct(channels, channels, 3, groups=channels)
        self.spatial_context = ConvNormAct(channels, channels, 3, dilation=2, groups=channels)
        self.spatial_fuse = ConvNormAct(channels * 2, channels, 1)
        self.frequency_projection = nn.Sequential(
            ConvNormAct(7, channels, 3),
            ConvNormAct(channels, channels, 3, groups=channels),
        )
        hidden = max(channels // 2, 8)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            ConvNormAct(channels * 2, channels, 1),
            ConvNormAct(channels, channels, 3, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, spatial, frequency_bands):
        low_frequency = frequency_bands[:, :1]
        details = frequency_bands[:, 1:]
        noise_level = details.abs().mean(dim=(2, 3), keepdim=True).detach()
        threshold = torch.sigmoid(self.threshold_logits) * noise_level
        clean_details = details.sign() * F.relu(details.abs() - threshold)
        frequency = torch.cat([low_frequency, clean_details, clean_details.abs()], dim=1)
        frequency = F.interpolate(frequency, size=spatial.shape[-2:], mode="bilinear", align_corners=False)

        spatial_feature = self.spatial_fuse(
            torch.cat([self.spatial_local(spatial), self.spatial_context(spatial)], dim=1)
        )
        frequency_feature = self.frequency_projection(frequency)
        gate = self.spatial_gate(torch.cat([spatial_feature, frequency_feature], dim=1))
        residual = self.residual(torch.cat([spatial_feature, gate * frequency_feature], dim=1))
        return spatial + torch.sigmoid(self.residual_scale) * residual


class LocalStatisticsSpeckleGuard(nn.Module):
    """LSSG: local-statistics gating for multiplicative speckle suppression."""

    def __init__(self, channels):
        super().__init__()
        self.coherent_detail = nn.Sequential(
            ConvNormAct(channels, channels, 3, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        hidden = max(channels // 4, 8)
        self.noise_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            ConvNormAct(channels * 2, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, feature, unused_frequency_bands=None):
        local_mean = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        residual = feature - local_mean
        local_variance = F.avg_pool2d(residual.square(), kernel_size=3, stride=1, padding=1)
        local_std = torch.sqrt(local_variance + 1e-5)
        standardized_detail = (residual / (local_std + 1e-5)).clamp(-3.0, 3.0)
        coherent_detail = self.coherent_detail(standardized_detail)
        gate = self.noise_gate(torch.cat([residual.abs(), local_std], dim=1))
        guarded = gate * local_mean + (1.0 - gate) * (feature + coherent_detail)
        correction = self.output(torch.cat([feature, guarded], dim=1))
        return feature + torch.sigmoid(self.residual_scale) * correction


class DepthwiseExpert(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size, dilation=dilation, groups=channels),
            ConvNormAct(channels, channels, 1),
        )

    def forward(self, x):
        return self.block(x)


class TriChallengeAdaptiveRouter(nn.Module):
    """Lightweight routing among noise, small-lesion, and boundary experts."""

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
        self.output = ResidualConvBlock(channels, channels)

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=1)
        experts = (
            self.noise_expert(x),
            self.small_lesion_expert(x),
            self.boundary_expert(x),
        )
        mixed = sum(expert * weights[:, index:index + 1] for index, expert in enumerate(experts))
        return self.output(x + mixed)


class ChannelGraphReasoning(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced = max(channels // 8, 16)
        self.reduce = nn.Conv2d(channels, reduced, 1, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feature = self.reduce(x)
        batch, channels, height, width = feature.shape
        values = feature.float().flatten(2)
        queries = F.normalize(values, dim=2, eps=1e-6)
        adjacency = torch.bmm(queries, queries.transpose(1, 2)) / math.sqrt(channels)
        adjacency = torch.softmax(adjacency, dim=-1)
        reasoned = torch.bmm(adjacency, values).view(batch, channels, height, width)
        reasoned = reasoned.to(dtype=x.dtype)
        return x + torch.tanh(self.gamma) * self.project(reasoned)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.projection = ConvNormAct(in_channels, out_channels, 3)
        self.fusion = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.projection(x)
        return self.fusion(torch.cat([skip, x], dim=1))


class BUSIOptUNet(nn.Module):
    def __init__(
        self,
        output_channels=1,
        base_channels=32,
        use_sfs=True,
        use_tricar=True,
        sfs_version=1,
        use_router=None,
        use_graph=None,
    ):
        super().__init__()
        channels = [base_channels * (2 ** index) for index in range(5)]
        self.use_sfs = use_sfs
        self.use_tricar = use_tricar
        use_router = use_tricar if use_router is None else use_router
        use_graph = use_tricar if use_graph is None else use_graph
        self.sfs_version = sfs_version
        self.haar = (
            HaarFrequencyDecomposition() if use_sfs and sfs_version == 2
            else None if use_sfs and sfs_version == 4
            else HaarHighFrequency() if use_sfs
            else None
        )

        self.encoders = nn.ModuleList(
            [
                ResidualConvBlock(3, channels[0]),
                ResidualConvBlock(channels[0], channels[1]),
                ResidualConvBlock(channels[1], channels[2]),
                ResidualConvBlock(channels[2], channels[3]),
                ResidualConvBlock(channels[3], channels[4]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        if use_sfs and sfs_version == 2:
            self.sfs_modules = nn.ModuleList(
                [SpeckleShrinkageSFSCalibration(channels[0]), SpeckleShrinkageSFSCalibration(channels[1])]
            )
        elif use_sfs and sfs_version == 4:
            self.sfs_modules = nn.ModuleList(
                [LocalStatisticsSpeckleGuard(channels[0]), LocalStatisticsSpeckleGuard(channels[1])]
            )
        elif use_sfs:
            self.sfs_modules = nn.ModuleList([SFSCalibration(channels[0]), SFSCalibration(channels[1])])
        else:
            self.sfs_modules = None
        self.challenge_routers = (
            nn.ModuleList([TriChallengeAdaptiveRouter(channel) for channel in channels])
            if use_router
            else None
        )
        self.graph_reasoning = ChannelGraphReasoning(channels[-1]) if use_graph else nn.Identity()
        self.decoders = nn.ModuleList(
            [
                UpBlock(channels[4], channels[3], channels[3]),
                UpBlock(channels[3], channels[2], channels[2]),
                UpBlock(channels[2], channels[1], channels[1]),
                UpBlock(channels[1], channels[0], channels[0]),
            ]
        )
        self.segmentation_head = nn.Conv2d(channels[0], output_channels, 1)

    def _route(self, feature, index):
        if self.challenge_routers is not None:
            return self.challenge_routers[index](feature)
        return feature

    def encode(self, image):
        high_frequency = self.haar(image) if self.haar is not None else None
        features = []
        x = image
        for index, encoder in enumerate(self.encoders):
            if index:
                x = self.pool(x)
            x = encoder(x)
            if self.sfs_modules is not None and index < len(self.sfs_modules):
                x = self.sfs_modules[index](x, high_frequency)
            x = self._route(x, index)
            features.append(x)
        features[-1] = self.graph_reasoning(features[-1])
        return features

    def decode(self, features):
        decoder_features = []
        x = features[-1]
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = decoder(x, skip)
            decoder_features.append(x)
        return x, decoder_features

    def forward_features(self, image):
        features = self.encode(image)
        decoded, decoder_features = self.decode(features)
        return decoded, features, decoder_features

    def forward(self, image):
        decoded, _, _ = self.forward_features(image)
        return self.segmentation_head(decoded)


def build_busi_opt_unet(variant="A3", base_channels=32):
    variants = {
        "A0": (False, False, False, 1),
        "A1": (True, False, False, 1),
        "A2": (False, True, True, 1),
        "A2R": (False, True, False, 1),
        "A2G": (False, False, True, 1),
        "A3": (True, True, True, 1),
        "A1V2": (True, False, False, 2),
        "A3V2": (True, True, True, 2),
        "A1V4": (True, False, False, 4),
        "A3V4": (True, True, True, 4),
    }
    if variant not in variants:
        raise ValueError("Unknown chapter-1 variant: {}".format(variant))
    use_sfs, use_router, use_graph, sfs_version = variants[variant]
    return BUSIOptUNet(
        base_channels=base_channels,
        use_sfs=use_sfs,
        use_tricar=use_router and use_graph,
        sfs_version=sfs_version,
        use_router=use_router,
        use_graph=use_graph,
    )
