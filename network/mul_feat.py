import torch
from torch import nn

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.img_processing import FrequencyDecomposer, NoiseGenerator, denormalize_batch_t
from utils.utils import init_layer

class MulFeat(nn.Module):
    def __init__(self, config):
        super(MulFeat, self).__init__()
        self.config = config
        self.rgb_backbone = create_backbone(config.model.feat.backbone, pretrained=config.model.pretrained, in_channels=3)
        self.freq_filter = config.model.feat.freq_filter
        self.freq_backbone = create_backbone(config.model.feat.backbone, pretrained=False, in_channels=3 * len(self.freq_filter))
        self.noise_backbone = create_backbone(config.model.feat.backbone, pretrained=False, in_channels=3)
        self.fd = FrequencyDecomposer()
        self.ng = NoiseGenerator()

    def init_weight(self):
        self.freq_backbone.apply(init_layer)
        self.noise_backbone.apply(init_layer)

    def forward(self, x):
        rgb_feat = self.rgb_backbone(x)
        freq_map = self.stack_frequency_map(x)
        freq_feat = self.freq_backbone(freq_map)
        noise_map = self.stack_noise_map(x)
        noise_feat = self.noise_backbone(noise_map)
        return rgb_feat, freq_feat, noise_feat

    def stack_frequency_map(self, img_t):
        if self.config.train.data_transforms.normalize_enabled:
            img_t = denormalize_batch_t(img_t, self.config.train.data_transforms.normalize.mean, self.config.train.data_transforms.normalize.std)
        freq_map = []
        for threshold in self.freq_filter:
            freq_map.append(self.fd.frequency_decomposition(img_t, threshold)[1])
        return torch.cat(freq_map, dim=1)

    def stack_noise_map(self, img_t):
        return self.ng.srm_generation(img_t)

def create_backbone(model, **kwargs):
    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }
    return model_dict[model](**kwargs)
