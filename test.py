import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from config import BaseConfig
from dataset import FolderDataset
from network.base_model import BaseModel
from utils.earlystop import EarlyStopping
from utils.img_processing import img_post_processing, denormalize_batch_t
from utils.metrics import eval_metrics
from utils.utils import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Test Argument Parser.')
    parser.add_argument('--config', type=str, default='configs/base_test.yaml', help='Configuration for Testing.')
    return parser.parse_args()


def prepare_test_output_dir(config, tag):
    output_dir = os.path.join(config.test.save_dir, tag)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir, 'logs')):
        os.makedirs(os.path.join(output_dir, 'logs'))
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
    parser = parse_args()
    config = BaseConfig(parser.config).cfg()
    print(config)

    set_random_seed(config.test.seed)
    test_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tag = 'test-{}'.format(test_time)
    output_dir = prepare_test_output_dir(config, tag)
    log_file = 'result.txt'
    # test_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    with open(os.path.join(output_dir, 'logs', 'config.txt') ,'w') as f:
        f.write(str(config))

    data_trans = transforms.Compose(img_trans(config))
    synthesized_dataset = FolderDataset(config.test.dataset.data_root, transform=data_trans)
    test_dataloader = DataLoader(dataset=synthesized_dataset, batch_size=1, shuffle=False, drop_last=False)

    device = 'cuda' if config.model.use_gpu else 'cpu'

    model = BaseModel(config)

    if config.model.load_from_checkpoint:
        model.load_model_state_dict(os.path.join(config.model.path, config.model.checkpoint_name))
        print('Successfully Load Checkpoint.')
    else:
        print('Model Checkpoint Required.')
        exit(0)

    probs, labels = torch.tensor([]).to(torch.device(device)), torch.tensor([]).to(torch.device(device))
    with tqdm(total=len(synthesized_dataset)) as test_pbar:
        test_pbar.set_description('Test Dataset Processing')
        for id, data in enumerate(test_dataloader):
            img_t, cls = data
            data_batch = cls.shape[0]
            img_t, cls = img_t.to(torch.device(device)), cls.to(torch.device(device))
            output, losses = model.test_batch((img_t, cls))
            pred = output
            prob = F.softmax(pred, dim=1)
            probs = torch.cat([probs, prob], dim=0)
            labels = torch.cat([labels, cls], dim=0)
            test_pbar.update(data_batch)
        # test_writer.add_scalar('loss', test_loss / test_num)
        probs = probs.cpu().detach()
        labels = labels.cpu().detach()
    result = eval_metrics(config.test.metrics, (probs.numpy(), labels.numpy()))

    with open(os.path.join(output_dir, 'logs', log_file), 'w') as log:
        log.write('Evaluation Metrics\n')
        for k, v in result.items():
            log.write('{}: {}\n'.format(k, v))
