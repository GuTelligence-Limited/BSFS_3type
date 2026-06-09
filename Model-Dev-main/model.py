import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling — learns the optimal pooling exponent p.
    Outperforms global average pooling on fine-grained recognition tasks.
    p=1 → average pooling, p→∞ → max pooling.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p)


class BSFSClassifier(nn.Module):
    """
    EfficientNet-B4 based classifier for Bristol Stool Form Scale (Type 1-7).

    Architecture:
    - EfficientNet-B4 pretrained backbone (IMAGENET1K_V1)
    - GeM pooling (better than avg pool for fine-grained tasks)
    - Classification head: Linear(1792→512) → BN → ReLU → Dropout(0.4)
                           → Linear(512→256) → BN → ReLU → Dropout(0.2)
                           → Linear(256→num_classes)

    EfficientNet-B4 features children (indices 0-8):
      0: Stem Conv2dNormActivation
      1: MBConv stage (depth 2)
      2: MBConv stage (depth 4)
      3: MBConv stage (depth 4)
      4: MBConv stage (depth 6)
      5: MBConv stage (depth 6)
      6: MBConv stage (depth 8)
      7: MBConv stage (depth 2)   ← unfreeze from here for Phase 2
      8: Final Conv2dNormActivation (1792ch)  ← Grad-CAM hook point

    Supports gradual unfreezing for 3-phase fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.4,
        use_gem: bool = True,
    ):
        super().__init__()
        self.use_gem = use_gem

        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # EfficientNet-B4.features is already a clean Sequential ending at 1792ch
        # No need to slice children — just use backbone.features directly
        self.features = backbone.features  # output: (B, 1792, H, W)

        if use_gem:
            self.pool = GeMPooling(p=3.0)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        in_features = 1792  # EfficientNet-B4 final channel count

        # Head: 1792 → 512 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B, 1792, H, W)
        x = self.pool(x)       # (B, 1792, 1, 1)
        x = self.classifier(x) # (B, num_classes)
        return x

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_from_layer: int = 0):
        """
        Gradually unfreeze backbone layers.

        EfficientNet-B4 features children (indices 0-8):
          0: Stem Conv
          1-7: MBConv stages
          8: Final Conv2dNormActivation (1792ch)

        For Phase 2 (last 2 blocks): unfreeze_from_layer=7
        For full fine-tune: use unfreeze_all()
        """
        children = list(self.features.children())
        for i, child in enumerate(children):
            requires_grad = (i >= unfreeze_from_layer)
            for param in child.parameters():
                param.requires_grad = requires_grad

    def unfreeze_all(self):
        """Unfreeze everything including GeM pool."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self):
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        print(f"Trainable params : {trainable:,} ({100 * trainable / total:.1f}%)")
        print(f"Total params     : {total:,}")
