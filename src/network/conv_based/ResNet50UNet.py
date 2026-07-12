import os

import torch
import torch.nn as nn
from torchvision.models import resnet50

from src.network.conv_based.BUSIOptUNet import ConvNormAct, ResidualConvBlock, UpBlock


DEFAULT_DEEPLAB_CHECKPOINT = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth"
)


class ResNet50UNet(nn.Module):
    """U-Net decoder on a locally cached COCO-pretrained ResNet50 encoder."""

    def __init__(
        self,
        output_channels=1,
        checkpoint_path=DEFAULT_DEEPLAB_CHECKPOINT,
        freeze_encoder_bn=False,
    ):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn
        encoder = resnet50(weights=None)
        self.encoder = encoder
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
                ConvNormAct(64, 32, 1),
                ConvNormAct(256, 64, 1),
                ConvNormAct(512, 128, 1),
                ConvNormAct(1024, 256, 1),
                ConvNormAct(2048, 512, 1),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                UpBlock(512, 256, 256),
                UpBlock(256, 128, 128),
                UpBlock(128, 64, 64),
                UpBlock(64, 32, 32),
                UpBlock(32, 32, 32),
            ]
        )
        self.segmentation_head = nn.Conv2d(32, output_channels, 1)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_encoder_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _load_deeplab_encoder(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Cached DeepLab checkpoint not found: {}".format(checkpoint_path)
            )
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = self.encoder.state_dict()
        compatible = {}
        for key, value in state.items():
            if not key.startswith("backbone."):
                continue
            encoder_key = key[len("backbone."):]
            if encoder_key in encoder_state and encoder_state[encoder_key].shape == value.shape:
                compatible[encoder_key] = value
        self.encoder.load_state_dict(compatible, strict=False)
        return len(compatible)

    def encode(self, image):
        full_resolution = self.input_skip(image)
        x = (image - self.image_mean) / self.image_std
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        features = [
            projection(feature)
            for projection, feature in zip(self.lateral, (stem, layer1, layer2, layer3, layer4))
        ]
        return full_resolution, features

    def decode(self, full_resolution, features):
        x = features[-1]
        decoder_features = []
        for decoder, skip in zip(self.decoders[:-1], reversed(features[:-1])):
            x = decoder(x, skip)
            decoder_features.append(x)
        x = self.decoders[-1](x, full_resolution)
        decoder_features.append(x)
        return x, decoder_features

    def forward_features(self, image):
        full_resolution, features = self.encode(image)
        decoded, decoder_features = self.decode(full_resolution, features)
        return decoded, features, decoder_features

    def forward(self, image):
        decoded, _, _ = self.forward_features(image)
        return self.segmentation_head(decoded)
