import torch
from torch import nn

from .identity import Identity
from utils.utils import init_layer
class FusionNeck(nn.Module):
    def __init__(self, config):
        super(FusionNeck, self).__init__()
        self.config = config
        self.feat_in_branches = 1    # rgb
        if config.model.feat.freq_branch_enabled:
            self.feat_in_branches += 1
        if config.model.feat.noise_branch_enabled:
            self.feat_in_branches += 1
        self.feat_in_channels = 512 * self.feat_in_branches
        self.fusion_out_channels = config.model.feat.fusion_out_channels
        # self.fusion_layer = nn.Sequential(
        #     nn.Conv2d(self.feat_in_channels, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, self.fusion_out_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.fusion_out_channels),
        #     nn.ReLU()
        # )
        self.fusion_layer = nn.Sequential(
            Identity()
        )
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def init_weight(self):
        self.apply(init_layer)

    def forward(self, x):
        """
        :param x: rgb_feat, freq_feat, noise_feat
        :return: feat, [B, C]
        """
        feat = torch.cat(x, dim=1)
        feat = self.fusion_layer(feat)
        feat = self.avg_pool2d(feat)
        feat = feat.view(feat.size(0), -1)
        return feat
