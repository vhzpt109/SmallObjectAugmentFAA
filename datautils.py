import torch

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, Sampler
from dataset import COCODataset, collate_fn


def get_dataloaders(dataroot, type='train', batch_size=8, fold_idx=0, augmentation=None):
    train_dataset = COCODataset(root=dataroot, split=type, augmentation=augmentation)

    sss = KFold(n_splits=4, shuffle=True, random_state=50)
    sss = sss.split(list(range(len(train_dataset))))
    for _ in range(fold_idx + 1):
        train_idx, valid_idx = next(sss)

    # train_sampler = SubsetRandomSampler(train_idx)
    train_sampler = SubsetSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=8, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    valid_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=valid_sampler, collate_fn=collate_fn)

    return train_data_loader, valid_data_loader


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
