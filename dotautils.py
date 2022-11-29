import torch

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from dataset import DOTADataset, collate_fn
from datautil import SubsetSampler


def get_dataloaders(dataroot, type='train', batch_size=8, fold_idx=0, augmentation=None):
    train_dataset = DOTADataset(root=dataroot, type=type, augmentation=augmentation)

    sss = KFold(n_splits=4, shuffle=True, random_state=50)
    sss = sss.split(list(range(len(train_dataset))))
    for _ in range(fold_idx):
        train_idx, valid_idx = next(sss)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=8, sampler=train_sampler, collate_fn=collate_fn)
    valid_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=valid_sampler, collate_fn=collate_fn)

    return train_data_loader, valid_data_loader


def get_valid_dataloaders(dataroot, type='val', batch_size=8):
    valid_dataset = DOTADataset(root=dataroot, type=type)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    return valid_data_loader