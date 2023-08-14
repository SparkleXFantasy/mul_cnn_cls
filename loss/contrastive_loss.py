import torch


class ContrastiveLoss:
    def __call__(self, pred, target, sign):
        """
        :param pred: [B, C, 1, 1]
        :param target: [B, C, 1, 1]
        :param sign: [B, 1]. The array for sample classes. 1 for positive samples, -1 for negative samples
        :return: The mean of N1 Loss of each pair
        """
        feat_l1 = torch.abs(pred - target)
        feat_l1 = torch.sum(feat_l1.view(feat_l1.shape[0], -1), dim=1) * sign
        feat_l1 = torch.mean(feat_l1)
        return feat_l1