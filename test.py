import cv2
import torch
import numpy as np

from dataset import COCODataset, collate_fn
from torchvision.utils import save_image


if __name__ == "__main__":
    dataroot = "/YDE/COCO"
    dataset = "COCO"
    model = "Faster R-CNN"
    # until = 5
    num_op = 2
    num_policy = 5
    num_search = 200
    cross_valid_num = 5
    cross_valid_ratio = 0.4
    num_epochs = 10
    num_classes = 91
    train_batch_size = 16
    valid_batch_size = 1

    # train_dataset = COCODataset(root=dataroot, split='train')
    valid_dataset = COCODataset(root=dataroot, split='val')

    # train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=train_batch_size, shuffle=True,
    #                                            num_workers=4)

    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=valid_batch_size,
                                                    collate_fn=collate_fn,
                                                    shuffle=False,
                                                    num_workers=2)

    print(len(valid_data_loader))

    device = torch.device('cuda:0')

    for i, (image, annotation) in enumerate(valid_data_loader):
        images = image
        print(annotation)
        boxes = annotation['boxes']
        masks = annotation['masks']
        labels = annotation['labels']
        area = annotation['area']

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        masks = [m.to(device) for m in masks]
        labels = [l.to(device) for l in labels]
        areas = [a.to(device) for a in area]
        print("boxes", boxes)
        # print("masks", masks[0][1].shape)
        image_test = images[0].cpu().numpy()
        print(image_test.shape)
        image_test = np.transpose(image_test, (1, 2, 0))
        # cv2.imwrite("test/" + str(i) + "_image.png", image_test)
        # cv2.imwrite("test/" + str(i) + "_mask.png", masks[0][0].cpu().numpy() * 255)
        print("labels", labels)
        print("area", area)