import os

import torch
import torch.nn as nn
from torchvision.models import resnet101

from src.network.conv_based.BUSIOptUNet import ConvNormAct, ResidualConvBlock, UpBlock
from src.network.conv_based.ResNet50UNet import ResNet50UNet


DEFAULT_DEEPLAB101_CHECKPOINT = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth"
)


class ResNet101UNet(ResNet50UNet):
    """U-Net decoder on a COCO-segmentation-pretrained ResNet101 encoder."""

    def __init__(self, output_channels=1, checkpoint_path=DEFAULT_DEEPLAB101_CHECKPOINT):
        nn.Module.__init__(self)
        self.encoder = resnet101(weights=None)
        self.loaded_encoder_tensors = 0
        if checkpoint_path:
            self.loaded_encoder_tensors = self._load_deeplab_encoder(checkpoint_path)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        self.input_skip = ResidualConvBlock(3, 32)
        self.lateral = nn.ModuleList(
            [
                ConvNormAct(64, 32, 1), ConvNormAct(256, 64, 1),
                ConvNormAct(512, 128, 1), ConvNormAct(1024, 256, 1),
                ConvNormAct(2048, 512, 1),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                UpBlock(512, 256, 256), UpBlock(256, 128, 128),
                UpBlock(128, 64, 64), UpBlock(64, 32, 32), UpBlock(32, 32, 32),
            ]
        )
        self.segmentation_head = nn.Conv2d(32, output_channels, 1)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self
