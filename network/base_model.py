import torch
from torch import nn, optim
from torch.nn import functional as F

from .mul_feat import MulFeat
from .fusion_neck import FusionNeck

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if config.model.use_gpu else 'cpu'
        self.feat_backbone = MulFeat(config)
        self.fusion_neck = FusionNeck(config)
        self.head = nn.Linear(config.model.feat.fusion_out_channels, config.model.head_cls)
        if self.config.model.use_gpu:
            self.deploy_gpu()
        self.CELoss = nn.CrossEntropyLoss()
        self.optim = optim.Adam([
            {'params': self.feat_backbone.parameters()},
            {'params': self.fusion_neck.parameters()},
            {'params': self.head.parameters()},
        ], lr=self.config.train.hyperparameter.learning_rate)

    def init_weight(self):
        self.feat_backbone.init_weight()
        self.fusion_neck.init_weight()
        nn.init.xavier_normal_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def save_model_state_dict(self, ckpt_path):
        state_dict = {
            'feat_backbone_state_dict': self.feat_backbone.state_dict(),
            'fusion_neck_state_dict': self.fusion_neck.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'optim_state_dict' : self.optim.state_dict(),
        }
        torch.save(state_dict, ckpt_path)
    
    def load_model_state_dict(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.feat_backbone.load_state_dict(state_dict['feat_backbone_state_dict'])
        self.fusion_neck.load_state_dict(state_dict['fusion_neck_state_dict'])
        self.head.load_state_dict(state_dict['head_state_dict'])
        self.optim.load_state_dict(state_dict['optim_state_dict'])

    def deploy_gpu(self):
        self.feat_backbone = self.feat_backbone.to(device=torch.device('cuda'))
        self.fusion_neck = self.fusion_neck.to(device=torch.device('cuda'))
        self.head = self.head.to(device=torch.device('cuda'))

    def train_batch(self, input):
        self.train()
        img_t, cls = input
        pred = self.forward(img_t)
        loss_ce = self.CELoss(pred, cls)
        loss = loss_ce
        loss.backward()
        self.optim.step()

        output = pred
        losses = {
            'TotalLoss': loss.item(),
            'CELoss': loss_ce.item(),
        }
        return output, losses
    
    def test_batch(self, input):
        self.eval()
        with torch.inference_mode():
            img_t, cls = input
            pred = self.forward(img_t)
            loss_ce = self.CELoss(pred, cls)
            loss = loss_ce
            output = pred
            losses = {
                'TotalLoss': loss.item(),
                'CELoss': loss_ce.item(),
            }
        return output, losses

    def forward(self, img_t):
        """
        :param img_t: [B, C, H, W]
        :return: pred [B, self.head_cls]
        """
        feat = self.feat_backbone(img_t)
        fusion_feat = self.fusion_neck(feat)
        pred = self.head(fusion_feat)
        return pred

    def train(self):
        self.feat_backbone.train()
        self.feat_backbone.train()
        self.head.train()

    def eval(self):
        self.feat_backbone.eval()
        self.fusion_neck.eval()
        self.head.eval()

    def get_cam_heatmap(self, img_t, cls_t):
        feat = self.feat_backbone(img_t)
        feat = torch.cat(feat, dim=1)
        fusion_featmap = self.fusion_neck.fusion_layer(feat)
        fc_weight = self.head.weight.data
        cls_weight = fc_weight[cls_t, :]
        cam_heatmap = (cls_weight.unsqueeze(-1).unsqueeze(-1) * fusion_featmap)    # broadcast the weight
        return cam_heatmap
