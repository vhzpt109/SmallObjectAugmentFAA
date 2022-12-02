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
from augmentations import SmallObjectAugmentation, ApplyFoundPolicy
from datautil import DOTALABELS
from dotautils import get_dataloaders

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

    final_policy_set = [[["SmallObjectAugmentMultiple", 0.01626266976413837, 0.49433010590138216]],
                        [["SmallObjectAugmentAll", 0.8042975847002628, 0.36636396190340426]],
                        [["SmallObjectAugmentMultiple", 0.5265840327492187, 0.10221705010856841]],
                        [["SmallObjectAugmentAll", 0.04983947291903776, 0.38160577136336227]],
                        [["SmallObjectAugmentOne", 0.09984886902535978, 0.7988124190873198]],
                        [["SmallObjectAugmentAll", 0.044906454938151597, 0.6333051724505026]],
                        [["SmallObjectAugmentOne", 0.03088111997730225, 0.733516768014754]],
                        [["SmallObjectAugmentOne", 0.3456079048573106, 0.8256234031567476]],
                        [["SmallObjectAugmentAll", 0.9629862138140098, 0.42765899921461953]],
                        [["SmallObjectAugmentMultiple", 0.065554557247403, 0.2266627671054462]],
                        [["SmallObjectAugmentOne", 0.4944232312536496, 0.797085349781581]],
                        [["SmallObjectAugmentAll", 0.03446246128095375, 0.5537226331580671]],
                        [["SmallObjectAugmentOne", 0.5610643206790065, 0.9165613300707536]],
                        [["SmallObjectAugmentAll", 0.1148301150200296, 0.7607161552798223]],
                        [["SmallObjectAugmentOne", 0.6805228937206661, 0.3780264498721946]],
                        [["SmallObjectAugmentOne", 0.4389568071759026, 0.4092837105614071]],
                        [["SmallObjectAugmentAll", 0.10730013033892644, 0.6597271197040204]],
                        [["SmallObjectAugmentOne", 0.8985023632203075, 0.9866860192127801]]]

    _, valid_data_loader = get_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=1, augmentation=[ApplyFoundPolicy(policies=final_policy_set)])

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

                        cv_image = cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                        cv_image = cv2.putText(cv_image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 0, 255), 1)
                    cv2.imwrite("visualization/" + str(count) + ".jpg", cv_image)

                    count += 1
