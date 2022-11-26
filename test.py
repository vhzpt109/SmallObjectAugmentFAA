import json
import os

import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import COCODataset, collate_fn
from torchvision.utils import save_image

from search import get_model_instance_segmentation, get_coco_stats

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
    valid_batch_size = 8

    valid_dataset = COCODataset(root=dataroot, split='val')
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=valid_batch_size,
                                                    collate_fn=collate_fn,
                                                    shuffle=False,
                                                    num_workers=4
                                                    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(num_classes=num_classes).to(device)

    model_path = "models/Mask_R-CNN_COCO_fold1.pth"
    # if exist model, evaluate model after load
    if os.path.exists(model_path):
        # print("%s Model Exist! Load Model.." % model_path)
        # checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        print('----------------------COCOeval Metric start--------------------------')
        inference_results = []
        count = 1
        for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
            print(i)
            with torch.no_grad():
                imgs = list(img.to(device=device, dtype=torch.float) for img in images_batch)
                # annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                inference = model(imgs)

                for batch_idx in range(len(images_batch)):
                    boxes, labels, scores, mask = inference[batch_idx]["boxes"], inference[batch_idx]["labels"].cpu(), \
                                                  inference[batch_idx]["scores"].cpu(), inference[batch_idx][
                                                      "masks"].cpu()

                    cv_image = imgs[batch_idx].detach().cpu().numpy()
                    cv_image = np.transpose(cv_image, (1, 2, 0))
                    cv_image = cv_image * 255.
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    cv_image = cv_image.astype(np.uint8).copy()

                    cats = valid_dataset.coco.loadCats(valid_dataset.coco.getCatIds())
                    nms = [cat['name'] for cat in cats]
                    insert_idx_list = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82]
                    for insert_idx in insert_idx_list:
                        nms.insert(insert_idx, "-")
                    nms.append("-")

                    for i in range(len(boxes)):
                        if scores[i] < 0.5:
                            continue
                        x_min, y_min, x_max, y_max = map(int, boxes[i])
                        label = labels[i]

                        cv_image = cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv_image = cv2.putText(cv_image, nms[label - 1], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                               1, (0, 0, 255), 2)
                    cv2.imwrite("visualization/" + str(count) + ".jpg", cv_image)

                    cv_mask = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), np.uint8)
                    for j in range(mask.shape[0]):
                        if scores[j] < 0.5:
                            continue
                        cv_masks = mask[j].detach().cpu().numpy()
                        cv_masks = cv_masks * 255
                        cv_masks = np.transpose(cv_masks, (1, 2, 0))
                        cv_masks = cv_masks.astype(np.uint8).copy()
                        cv_mask = cv2.bitwise_or(cv_mask, cv_masks[:, :, 0])
                    cv2.imwrite("visualization/" + str(count) + "_masks.jpg", cv_mask)

                    count += 1

        print('----------------------COCOeval Metric end--------------------------')
