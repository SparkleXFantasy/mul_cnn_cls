from torch.utils.data import random_split

from .folder_dataset import FolderDataset

__all__ = [
    'FolderDataset', 'random_split_dataset'
]


def random_split_dataset(dataset, ratio):
    try:
        eps = 1e-6
        assert(sum(ratio) - 1.0 < eps)
    except:
        print('The sum of expected ratio for split should be 1.0 in total. Got {}.'.format(sum(ratio)))
        exit(0)
    total_num = len(dataset)
    split_lengths = [int(total_num * ratio[i]) for i in range(len(ratio) - 1)]
    split_lengths.append(total_num - sum(split_lengths))
    return random_split(dataset, split_lengths)