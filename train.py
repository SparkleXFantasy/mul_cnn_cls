import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score, roc_auc_score
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import BaseConfig
from dataset import FolderDataset, random_split_dataset
from network.base_model import BaseModel
from utils.earlystop import EarlyStopping
from utils.img_processing import img_post_processing, denormalize_batch_t
from utils.metrics import eval_metrics
from utils.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Argument Parser.')
    parser.add_argument('--config', type=str, default='configs/base_train.yaml', help='Configuration for Training.')
    return parser.parse_args()


def prepare_train_output_dir(config, tag):
    output_dir = os.path.join(config.train.save_dir, tag)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir, 'models')):
        os.makedirs(os.path.join(output_dir, 'models'))
    return output_dir


def img_trans(config):
    data_config = config.train.data_transforms
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

    set_random_seed(config.train.seed)
    train_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tag = 'train-{}'.format(train_time)
    output_dir = prepare_train_output_dir(config, tag)

    train_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', 'test'))
    val_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', 'val'))
    with open(os.path.join(output_dir, 'logs', 'config.txt') ,'w') as f:
        f.write(str(config))

    data_trans = transforms.Compose(img_trans(config))
    synthesized_dataset = FolderDataset(config.train.dataset.data_root, transform=data_trans)
    train_dataset, val_dataset, test_dataset = random_split_dataset(synthesized_dataset, config.train.dataset.split)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train.dataset.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, drop_last=False)
    device = 'cuda' if config.model.use_gpu else 'cpu'

    model = BaseModel(config)
    if config.model.load_from_checkpoint:
        model.load_model_state_dict(os.path.join(config.model.path, config.model.checkpoint_name))
    else:
        model.init_weight()

    early_stop = EarlyStopping(config.train.hyperparameter.early_stop, delta=0.001, verbose=True)
    early_stop_enabled = config.train.hyperparameter.early_stop_enabled

    for epoch in range(config.train.epoch):
        train_loss, train_num = 0, 0
        val_loss, val_num = 0, 0

        train_probs, train_labels = torch.tensor([]).to(torch.device(device)), torch.tensor([]).to(torch.device(device))
        with tqdm(total=len(train_dataset)) as train_pbar:
            train_pbar.set_description('Training Data Processing')
            for id, data in enumerate(train_dataloader):
                img_t, cls = data
                data_batch = cls.shape[0]
                train_num += data_batch
                if config.model.use_gpu:
                    img_t = img_t.to(torch.device(device))
                    cls = cls.to(torch.device(device))
                output, losses = model.train_batch((img_t, cls))
                pred = output
                prob = F.softmax(pred, dim=1)
                train_probs = torch.cat([train_probs, prob], dim=0)
                train_labels = torch.cat([train_labels, cls], dim=0)
                train_loss += losses['TotalLoss'] * data_batch
                train_pbar.update(data_batch)
            train_probs = train_probs.cpu().detach()
            train_labels = train_labels.cpu().detach()
        train_metrics = eval_metrics(config.train.metrics, (train_probs.numpy(), train_labels.numpy()))
        print('Training Loss: {}.'.format(train_loss / train_num))

        val_probs, val_labels = torch.tensor([]).to(torch.device(device)), torch.tensor([]).to(torch.device(device))
        with tqdm(total=len(val_dataset)) as val_pbar:
            val_pbar.set_description('Validation Data Processing')
            for id, data in enumerate(val_dataloader):
                img_t, cls = data
                data_batch = cls.shape[0]
                val_num += data_batch
                if config.model.use_gpu:
                    img_t = img_t.to(torch.device(device))
                    cls = cls.to(torch.device(device))
                output, losses = model.test_batch((img_t, cls))
                pred = output
                prob = F.softmax(pred, dim=1)
                val_probs = torch.cat([val_probs, prob], dim=0)
                val_labels = torch.cat([val_labels, cls], dim=0)
                val_loss += losses['TotalLoss'] * data_batch
                val_pbar.update(data_batch)
            val_probs = val_probs.cpu().detach()
            val_labels = val_labels.cpu().detach()
        val_metrics = eval_metrics(config.train.metrics, (val_probs.numpy(), val_labels.numpy()))
        print('Validation Loss: {}.'.format(val_loss / val_num))

        if early_stop_enabled:
            early_stop(-val_loss, model, os.path.join(output_dir, 'models', '{}_best.pth'.format(config.train.save_name)))

        if (epoch + 1) % config.train.save_step == 0:
            model.save_model_state_dict(os.path.join(output_dir, 'models', '{}_{}.pth'.format(config.train.save_name, epoch)))

        if (epoch + 1) % config.train.log_step == 0 or early_stop.early_stop:
            train_writer.add_scalar('loss', train_loss / train_num, global_step=epoch)
            val_writer.add_scalar('loss', val_loss / val_num, global_step=epoch)
            print('Training Metrics')
            for k, v in train_metrics.items():
                train_writer.add_scalar(k, v, global_step=epoch)
                print('{}: {}'.format(k, v))
            print('Validation Metrics')
            for k, v in val_metrics.items():
                val_writer.add_scalar(k, v, global_step=epoch)
                print('{}: {}'.format(k, v))

            test_loss, test_num = 0, 0
            test_probs, test_labels = torch.tensor([]).to(torch.device(device)), torch.tensor([]).to(
                torch.device(device))
            with tqdm(total=len(test_dataset)) as test_pbar:
                test_pbar.set_description('Test Dataset Processing')
                for id, data in enumerate(test_dataloader):
                    img_t, cls = data
                    data_batch = cls.shape[0]
                    test_num += data_batch
                    if config.model.use_gpu:
                        img_t = img_t.to(torch.device(device))
                        cls = cls.to(torch.device(device))
                    output, losses = model.test_batch((img_t, cls))
                    pred = output
                    prob = F.softmax(pred, dim=1)
                    test_probs = torch.cat([test_probs, prob], dim=0)
                    test_labels = torch.cat([test_labels, cls], dim=0)
                    test_loss += losses['TotalLoss'] * data_batch
                    test_pbar.update(data_batch)
                test_probs = test_probs.cpu().detach()
                test_labels = test_labels.cpu().detach()
            test_metrics = eval_metrics(config.train.metrics, (test_probs.numpy(), test_labels.numpy()))
            test_writer.add_scalar('loss', test_loss / test_num, global_step=epoch)
            print('Testing Metrics')
            for k, v in test_metrics.items():
                test_writer.add_scalar(k, v, global_step=epoch)
                print('{}: {}'.format(k, v))
            print('Test Loss: {}.'.format(test_loss / test_num))

        print('Epoch: {0} / {1}.'.format(epoch, config.train.epoch))

        if early_stop_enabled and early_stop.early_stop:
            print('Early Stopping.')
            break
