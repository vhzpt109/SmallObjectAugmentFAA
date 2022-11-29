import torch

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from dataset import COCODataset, collate_fn
from datautil import SubsetSampler


def get_dataloaders(dataroot, type='train', batch_size=8, fold_idx=0, augmentation=None):
    train_dataset = COCODataset(root=dataroot, type=type, augmentation=augmentation)

    sss = KFold(n_splits=4, shuffle=True, random_state=50)
    sss = sss.split(list(range(len(train_dataset))))
    for _ in range(fold_idx):
        train_idx, valid_idx = next(sss)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=8, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    valid_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=valid_sampler, collate_fn=collate_fn)

    return train_data_loader, valid_data_loader


def get_valid_dataloaders(dataroot, type='val', batch_size=8):
    valid_dataset = COCODataset(root=dataroot, type=type)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    return valid_data_loader


def get_coco_stats(coco_stats, is_print=False):
    AP_IoU05_95_area_all_maxDets100, AP_IoU05_area_all_maxDets100, AP_IoU075_area_all_maxDets100, AP_IoU05_95_area_small_maxDets100, \
    AP_IoU05_95_area_medium_maxDets100, AP_IoU05_95_area_large_maxDets100, AR_IoU05_95_area_all_maxDets1, AR_IoU05_95_area_all_maxDets10, \
    AR_IoU05_95_area_all_maxDets100, AR_IoU05_95_area_small_maxDets100, AR_IoU05_95_area_medium_maxDets100, AR_IoU05_95_area_large_maxDets100 \
        = map(float, coco_stats)

    if is_print:
        print(f"AP_IoU05_95_area_all_maxDets100: {AP_IoU05_95_area_all_maxDets100:.5f}, AP_IoU05_area_all_maxDets100: {AP_IoU05_area_all_maxDets100:.5f}, "
              f"AP_IoU075_area_all_maxDets100: {AP_IoU075_area_all_maxDets100:.5f}, AP_IoU05_95_area_small_maxDets100: {AP_IoU05_95_area_small_maxDets100:.5f}, "
              f"AP_IoU05_95_area_medium_maxDets100: {AP_IoU05_95_area_medium_maxDets100:.5f}, AP_IoU05_95_area_large_maxDets100: {AP_IoU05_95_area_large_maxDets100:.5f},"
              f"AR_IoU05_95_area_all_maxDets1: {AR_IoU05_95_area_all_maxDets1:.5f}, AR_IoU05_95_area_all_maxDets10: {AR_IoU05_95_area_all_maxDets10:.5f},"
              f"AR_IoU05_95_area_all_maxDets100: {AR_IoU05_95_area_all_maxDets100:.5f}, AR_IoU05_95_area_small_maxDets100: {AR_IoU05_95_area_small_maxDets100:.5f},"
              f"AR_IoU05_95_area_medium_maxDets100: {AR_IoU05_95_area_medium_maxDets100:.5f}, AR_IoU05_95_area_large_maxDets100: {AR_IoU05_95_area_large_maxDets100:.5f}")

    result = {"AP_IoU05_95_area_all_maxDets100": AP_IoU05_95_area_all_maxDets100,
              "AP_IoU05_area_all_maxDets100": AP_IoU05_area_all_maxDets100,
              "AP_IoU075_area_all_maxDets100": AP_IoU075_area_all_maxDets100,
              "AP_IoU05_95_area_small_maxDets100": AP_IoU05_95_area_small_maxDets100,
              "AP_IoU05_95_area_medium_maxDets100": AP_IoU05_95_area_medium_maxDets100,
              "AP_IoU05_95_area_large_maxDets100": AP_IoU05_95_area_large_maxDets100,
              "AR_IoU05_95_area_all_maxDets1": AR_IoU05_95_area_all_maxDets1,
              "AR_IoU05_95_area_all_maxDets10": AR_IoU05_95_area_all_maxDets10,
              "AR_IoU05_95_area_all_maxDets100": AR_IoU05_95_area_all_maxDets100,
              "AR_IoU05_95_area_small_maxDets100": AR_IoU05_95_area_small_maxDets100,
              "AR_IoU05_95_area_medium_maxDets100": AR_IoU05_95_area_medium_maxDets100,
              "AR_IoU05_95_area_large_maxDets100": AR_IoU05_95_area_large_maxDets100}

    return result
