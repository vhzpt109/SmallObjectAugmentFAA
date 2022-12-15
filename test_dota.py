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
from dotautils import get_kfold_dataloaders, get_dataloaders

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
    num_classes = 19
    batch_size = 16

    final_policy_set = [[["SmallObjectAugmentMultiple", 0.1120818476673879, 0.3258170652459429]],
                        [["SmallObjectAugmentAll", 0.571990054889772, 0.9464573353634441]],
                        [["SmallObjectAugmentOne", 0.12896357750039, 0.565134349316077]],
                        [["SmallObjectAugmentOne", 0.13054025874016983, 0.8125871152053538]],
                        [["SmallObjectAugmentAll", 0.11702076163783126, 0.9392151173852]],
                        [["SmallObjectAugmentOne", 0.18134874058113715, 0.42480348475570184]],
                        [["SmallObjectAugmentAll", 0.9294191751009471, 0.5995885017281503]],
                        [["SmallObjectAugmentMultiple", 0.18675268578970416, 0.8907562627184183]],
                        [["SmallObjectAugmentOne", 0.7451604714897108, 0.4939563479816229]],
                        [["SmallObjectAugmentMultiple", 0.15846846770306672, 0.3702079390000968]],
                        [["SmallObjectAugmentMultiple", 0.05585314564912615, 0.25065554798291234]],
                        [["SmallObjectAugmentOne", 0.3669738533291767, 0.2389693045847482]],
                        [["SmallObjectAugmentMultiple", 0.08818137078698213, 0.2840562941798406]],
                        [["SmallObjectAugmentOne", 0.3422591967937419, 0.15419423177456063]],
                        [["SmallObjectAugmentMultiple", 0.42868228124301877, 0.5143922327635384]],
                        [["SmallObjectAugmentOne", 0.03843655731598017, 0.7987590418123129]],
                        [["SmallObjectAugmentAll", 0.19068739108723387, 0.7475272572599441]],
                        [["SmallObjectAugmentMultiple", 0.0928307835252018, 0.45897577282635726]],
                        [["SmallObjectAugmentMultiple", 0.036179592834423034, 0.9614435804303715]],
                        [["SmallObjectAugmentAll", 0.9909753301035339, 0.30101589310141996]],
                        [["SmallObjectAugmentMultiple", 0.0920874585542843, 0.9329892358102304]],
                        [["SmallObjectAugmentAll", 0.3987362483436294, 0.41845559163565615]],
                        [["SmallObjectAugmentMultiple", 0.07752224356416786, 0.7554302659160679]],
                        [["SmallObjectAugmentAll", 0.28665997110128105, 0.37348829946500867]],
                        [["SmallObjectAugmentMultiple", 0.13142426491279646, 0.9924136216187245]],
                        [["SmallObjectAugmentAll", 0.9814841703957331, 0.23825076858068267]],
                        [["SmallObjectAugmentMultiple", 0.34253709670116195, 0.45921535923503576]],
                        [["SmallObjectAugmentOne", 0.20208435665056812, 0.7304785742118938]],
                        [["SmallObjectAugmentOne", 0.4270846552134828, 0.3575644446137237]]]

    # _, valid_data_loader = get_kfold_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=1, augmentation=[ApplyFoundPolicy(policies=final_policy_set)])
    # _, valid_data_loader = get_kfold_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=1, augmentation=None)
    valid_data_loader = get_dataloaders(dataroot=dataroot, type='val', batch_size=batch_size, augmentation=None)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = getFasterRCNN(num_classes=num_classes).to(device)

    model_path = "models_faster_r-cnn/Faster_R-CNN_optimal_augment_fold3.pth"
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
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                inference = model(imgs)

                for batch_idx in range(len(images_batch)):
                    boxes, labels = annotations[batch_idx]["boxes"], annotations[batch_idx]["labels"].cpu(),

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
                    cv2.imwrite("visualization/" + str(count) + "_annot.jpg", cv_image)

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
