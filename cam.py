import argparse
import numpy as np
import os
import torch

from datetime import datetime
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import BaseConfig
from dataset import FolderDataset
from network.base_model import BaseModel
from utils.img_processing import denormalize_batch_t, img_post_processing
from utils.utils import set_random_seed, grayscale_to_color

def parse_args():
    parser = argparse.ArgumentParser(description='CAM Argument Parser.')
    parser.add_argument('--config', type=str, default='configs/base_test.yaml', help='Configuration for CAM testing.')
    return parser.parse_args()


def prepare_test_output_dir(config, tag):
    output_dir = os.path.join(config.test.save_dir, tag)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir, 'logs')):
        os.makedirs(os.path.join(output_dir, 'logs'))
    if not os.path.isdir(os.path.join(output_dir, 'visual')):
        os.makedirs(os.path.join(output_dir, 'visual'))
    return output_dir


def img_trans(config):
    data_config = config.test.data_transforms
    trans = []
    if data_config.crop_enabled:
        trans.append(transforms.Resize(data_config.resize))
        trans.append(transforms.RandomApply(nn.ModuleList([transforms.RandomCrop(data_config.crop)]), p=0.5))
    if data_config.flip_enabled:
        trans.append(transforms.RandomHorizontalFlip(p=0.5))
    if data_config.post_processing_enabled:
        trans.append(transforms.Lambda(lambda img: img_post_processing(img, data_config)))
    if data_config.resize_enabled:
        trans.append(transforms.Resize(data_config.resize))
    trans.append(transforms.ToTensor())
    if data_config.normalize_enabled:
        trans.append(transforms.Normalize(mean=data_config.normalize.mean, std=data_config.normalize.std))
    return trans


if __name__ == '__main__':
    args = parse_args()
    config = BaseConfig(args.config).cfg()
    print(config)

    set_random_seed(config.test.seed)

    run_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tag = 'cam-{}'.format(run_time)
    output_dir = prepare_test_output_dir(config, tag)

    with open(os.path.join(output_dir, 'logs', 'config.txt') ,'w') as f:
        f.write(str(config))

    data_trans = transforms.Compose(img_trans(config))
    synthesized_dataset = FolderDataset(config.test.dataset.data_root, transform=data_trans)
    test_dataloader = DataLoader(dataset=synthesized_dataset, batch_size=1, shuffle=True, drop_last=False)

    device = 'cuda' if config.model.use_gpu else 'cpu'

    model = BaseModel(config)
    model.eval()
    try:
        assert(config.model.load_from_checkpoint)
        model.load_model_state_dict(os.path.join(config.model.path, config.model.checkpoint_name))
        print('Successfully Load Checkpoint.')
    except:
        print('Model Checkpoint Required.')
        exit(0)

    channel_ratio = torch.zeros((2, 3), dtype=torch.float)
    cls_counter = torch.zeros((2), dtype=torch.int)
    img_counter = 0
    with torch.inference_mode():
        with tqdm(total=len(synthesized_dataset)) as process_pbar:
            process_pbar.set_description('Dataset Processing')
            for id, data in enumerate(test_dataloader):
                img_t, cls = data
                img_t, cls = img_t.to(torch.device(device)), cls.to(torch.device(device))
                batch_size = cls.shape[0]
                cam_heatmap = model.get_cam_heatmap(img_t, cls).cpu()    # [B, C, H, W]
                channel_heatmaps = torch.zeros((3), dtype=torch.float)
                for i in range(3):
                    avgpool_heatmap = F.adaptive_avg_pool2d(cam_heatmap, (1, 1))
                    channel_heatmaps[i] = (torch.sum(avgpool_heatmap[:, 512 * i: 512 * (i + 1), :, :], dim=(1, 2, 3)))    # each domain produces features with 512 channels
                for c in cls:
                    cls_counter[c] += 1
                    channel_ratio[c, :] += torch.abs(channel_heatmaps)
                total_heatmap = F.relu(torch.sum(cam_heatmap, dim=1))
                if config.test.data_transforms.normalize_enabled:
                    img_t = denormalize_batch_t(img_t, mean=config.test.data_transforms.normalize.mean, std=config.test.data_transforms.normalize.std)
                for i in range(batch_size):
                    img = 255. * np.transpose(img_t[i].cpu().numpy(), (1, 2, 0))
                    cover = Image.fromarray(img.astype(np.uint8))
                    color_heatmap = grayscale_to_color(total_heatmap[i].numpy())
                    visual_heatmap = Image.fromarray(color_heatmap).resize(cover.size)
                    blended_map = Image.blend(cover, visual_heatmap, 0.4)
                    if cls[i] == 1:
                        blended_map.save(os.path.join(output_dir, 'visual', '{}_{}.png'.format(cls[i], img_counter)))
                        img_counter += 1
            normalized_channel_ratio = F.normalize(channel_ratio, p=1, dim=1)
            with open(os.path.join(output_dir, 'logs', 'result.txt'), 'w') as f:
                f.write('Channel Contribution for Classes.\n')
                for i in range(normalized_channel_ratio.shape[0]):
                    f.write('Class {0}: {1:.4f}, {2:.4f}, {3:.4f}\n'.format(i, normalized_channel_ratio[i, 0], normalized_channel_ratio[i, 1], normalized_channel_ratio[i, 2]))
