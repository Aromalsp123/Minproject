import torch
import torch.nn as nn
import timm


class MicroFormerX(nn.Module):
    def __init__(self, num_classes=3, num_sources=3):
        super(MicroFormerX, self).__init__()

        # Swin Tiny backbone
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )

        # ConvNeXt Tiny backbone
        self.convnext = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=0
        )

        swin_dim = self.swin.num_features
        conv_dim = self.convnext.num_features
        fusion_dim = swin_dim + conv_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.morph_head = nn.Linear(512, num_classes)
        self.source_head = nn.Linear(512, num_sources)

    def forward(self, x):
        f1 = self.swin(x)
        f2 = self.convnext(x)

        f = torch.cat([f1, f2], dim=1)
        f = self.fusion(f)

        morph = self.morph_head(f)
        source = self.source_head(f)

        return morph, source, f