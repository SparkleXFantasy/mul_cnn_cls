import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import FolderDataset
from utils.img_processing import NoiseGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Test Argument Parser.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='the path of a folder dataset.')
    parser.add_argument('-o', '--output', type=str, default='output', help='the output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ng = NoiseGenerator()
    dataset = FolderDataset(args.dataset)
    batch_size = 1
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    device = 'cuda'

    length = len(dataset)
    # data_subclass_num = [0] * len(dataset.classes)
    # total_noise = [[]] * len(dataset.classes)
    with tqdm(total=length) as pbar:
        for id, data in enumerate(dataloader):
            img_t, cls = data
            img_t, cls = img_t.to(torch.device(device)), cls.to(torch.device(device))
            noise = ng.srm_generation(img_t)
            norm_noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
            save_image(norm_noise, 'output/{}_{}.png'.format(cls.cpu().item(), id))
            pbar.update(batch_size)