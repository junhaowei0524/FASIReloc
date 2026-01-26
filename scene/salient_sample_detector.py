import torch
import torch.nn as nn
import torch.nn.functional as F


def simple_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))

    res = torch.where(max_mask, scores, zeros)
    return res


class SalientSampleDetector(nn.Module):
    def __init__(self, in_channels, base_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),  # [B,C=base_channels,H,W]
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),  # [B,2C,H/2,W/2]
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),  # [B,4C,H/4,W/4]
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # [B,2C,H/2,W/2]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # [B,C,H,W]
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(base_channels, 1, 1)

        self.feature_head = nn.Conv2d(base_channels, 128, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        feat = self.encoder(x)
        feat = self.decoder(feat)

        heatmap = torch.sigmoid(self.heatmap_head(feat))
        heatmap = heatmap.squeeze(0)
        feature_map = F.normalize(self.feature_head(feat), p=2, dim=1)

        return heatmap
