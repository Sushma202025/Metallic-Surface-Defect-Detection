"""
model.py — CNN + ViT Hybrid for Metallic Surface Defect Detection

Accepts img_size parameter so training can use 112x112 (fast CPU mode)
or 224x224 (full accuracy mode) without code changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CNNFeatureExtractor(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


class CNN_ViT(nn.Module):
    """
    Hybrid CNN + ViT model.

    Args:
        num_classes (int):   Number of output classes.
        dropout     (float): Dropout in CNN blocks.
        pretrained  (bool):  Load pretrained ViT weights.
        img_size    (int):   Input image size. Use 112 for fast CPU training,
                             224 for full accuracy. Default: 224.
    """

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float   = 0.1,
        pretrained: bool = True,
        img_size: int    = 224,
    ):
        super().__init__()
        self.img_size = img_size

        self.cnn = CNNFeatureExtractor(dropout=dropout)

        # 1×1 conv: 64 channels → 3 channels (so ViT patch embedding works)
        self.adapter = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )

        # ViT always expects 224×224 internally — we upsample to that
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)          # (B, 64, H/4, W/4)
        x = self.adapter(x)      # (B, 3,  H/4, W/4)
        # Always upsample to 224×224 so ViT patch embedding is consistent
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vit(x)

    def freeze_vit_backbone(self):
        """Freeze all ViT layers except the final classification head."""
        for name, param in self.vit.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
