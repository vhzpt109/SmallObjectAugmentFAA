import torch

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from dataset import DOTADataset, collate_fn
from datautil import SubsetSampler


def get_kfold_dataloaders(dataroot, type='train', batch_size=8, fold_idx=0, augmentation=None):
    train_dataset = DOTADataset(root=dataroot, type=type, augmentation=augmentation)

    sss = KFold(n_splits=4, shuffle=True, random_state=50)
    sss = sss.split(list(range(len(train_dataset))))
    for _ in range(fold_idx):
        train_idx, valid_idx = next(sss)

    train_sampler = SubsetSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    valid_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=valid_sampler, collate_fn=collate_fn, pin_memory=True)

    return train_data_loader, valid_data_loader


def get_dataloaders(dataroot, type='train', batch_size=8, augmentation=None):
    dataset = DOTADataset(root=dataroot, type=type, augmentation=augmentation)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    return data_loader
