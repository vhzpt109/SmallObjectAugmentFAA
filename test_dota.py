import json
import os

import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations

from dataset import DOTADataset, collate_fn
from torchvision.utils import save_image
from augmentations import SmallObjectAugmentation
from datautil import DOTALABELS

from models import getFasterRCNN

if __name__ == "__main__":
    dataroot = '/YDE/DOTA/split_ss_dota'
    dataset = "DOTA"
    model = "Faster R-CNN"
    # until = 5
    num_op = 2
    num_policy = 5
    num_search = 200
    cross_valid_num = 5
    cross_valid_ratio = 0.4
    num_epochs = 10
    num_classes = 18
    batch_size = 16

    # augment = albumentations.Compose([
    #     SmallObjectAugmentation(copy_times=1, one_object=True, p=1.),
    #     SmallObjectAugmentation(copy_times=1, p=1.),
    #     SmallObjectAugmentation(copy_times=1, all_objects=True, p=1.)
    # ])

    valid_dataset = DOTADataset(root=dataroot, type='val', augmentation=None)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=num_classes,
                                                    collate_fn=collate_fn,
                                                    shuffle=False,
                                                    num_workers=4
                                                    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = getFasterRCNN(num_classes=num_classes).to(device)

    model_path = "models/Faster_R-CNN_DOTA_fold1.pth"
    # if exist model, evaluate model after load
    if os.path.exists(model_path):
        print("%s Model Exist! Load Model.." % model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        inference_results = []
        count = 1
        for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
            print(i)
            with torch.no_grad():
                imgs = list(img.to(device) for img in images_batch)
                # annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                inference = model(imgs)

                for batch_idx in range(len(images_batch)):
                    boxes, labels, scores = inference[batch_idx]["boxes"], inference[batch_idx]["labels"].cpu(), inference[batch_idx]["scores"].cpu()

                    cv_image = imgs[batch_idx].detach().cpu().numpy()
                    cv_image = np.transpose(cv_image, (1, 2, 0))
                    cv_image = cv_image * 255.
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    cv_image = cv_image.astype(np.uint8).copy()

                    values = {v: k for k, v in DOTALABELS.items()}  # // {'AA': '0', 'BB': '1', 'CC': '2'}

                    for i in range(len(boxes)):
                        x_min, y_min, x_max, y_max = map(int, boxes[i])
                        label = values.get(labels[i].item())

                        cv_image = cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv_image = cv2.putText(cv_image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 0, 255), 2)
                    cv2.imwrite("visualization/" + str(count) + ".jpg", cv_image)

                    count += 1
