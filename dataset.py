import os
import torch
import cv2
import numpy as np
import albumentations
import albumentations.pytorch
import glob
import time

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from datautil import DOTALABELS


def collate_fn(batch):
    return tuple(zip(*batch))


class DOTADataset(Dataset):
    def __init__(self,
                 root='/YDE/DOTA/split_ss_dota',
                 type='train',
                 augmentation=None,
                 save_visualization=False):
        super().__init__()

        self.root = root

        assert type in ['train', 'val', 'test']
        self.type = type

        self.augmentation = augmentation

        self.augmentation = augmentation

        self.save_visualization = save_visualization

        self.img_path = os.path.join(self.root, self.type, 'images')
        self.ann_path = os.path.join(self.root, self.type, 'annfiles')

        self.img_list = os.listdir(self.img_path)
        self.img_list.sort()

    def __getitem__(self, index):
        if index >= len(self.img_list):
            raise StopIteration

        image = np.array(Image.open(self.img_path + '/' + self.img_list[index]).convert('RGB'))
        h, w = image.shape[0], image.shape[1]

        ann_file_path = self.ann_path + '/' + self.img_list[index][:-4] + ".txt"
        ann_file = open(ann_file_path, "r")

        boxes = []
        labels = []
        difficult = []

        annotations = ann_file.readlines()
        for annotation in annotations:
            split_annotation = annotation.split(" ")
            boxes.append([split_annotation[0], split_annotation[1], split_annotation[4], split_annotation[5]])
            labels.append(DOTALABELS[split_annotation[8]])
            difficult.append(split_annotation[9].rstrip())

        boxes = np.array(boxes, dtype=np.float32)

        try:
            np.testing.assert_equal(len(boxes) > 0, True)  # check error occurring bbox
            np.testing.assert_equal(np.all([boxes[:, :] > 0]), True)  # check error occurring bbox
            np.testing.assert_equal(np.all([boxes[:, :] < w, boxes[:, :] < h]), True)  # check error occurring bbox
        except AssertionError:
            del self.img_list[index]
            return self.__getitem__(index)

        labels = np.array(labels, dtype=np.int64)
        difficult = np.array(difficult, dtype=np.int32)

        self.albumentation_transforms = albumentations.Compose([
            albumentations.pytorch.ToTensorV2()
        ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=["labels"]))

        if self.augmentation is not None:
            for augmentation in self.augmentation:
                self.albumentation_transforms.transforms.insert(0, augmentation)

        augmented = self.albumentation_transforms(image=image, bboxes=boxes, labels=labels)
        image = augmented["image"]
        boxes = augmented["bboxes"]
        labels = augmented["labels"]

        if self.save_visualization:
            cv_image = image.detach().cpu().numpy()
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
            cv2.imwrite("visualization/" + self.img_list[index], cv_image)

        image = torch.as_tensor(image / 255., dtype=torch.float32)
        result_annotation = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'difficult': torch.as_tensor(difficult, dtype=torch.int32),
        }

        return image, result_annotation

    def __len__(self):
        return len(self.img_list)


class COCODataset(Dataset):
    def __init__(self,
                 root='/YDE/COCO',
                 type='train',
                 augmentation=None,
                 save_visualization=False):
        super().__init__()

        self.root = root

        assert type in ['train', 'val', 'test']
        self.split = type
        self.set_name = type + '2017'

        self.augmentation = augmentation

        self.save_visualization = save_visualization

        self.img_path = glob.glob(os.path.join(self.root, 'images', self.set_name, '*.jpg'))
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_' + self.set_name + '.json'))

        self.img_id = list(self.coco.imgToAnns.keys())

        self.coco_ids = sorted(self.coco.getCatIds())  # list of coco labels [1, ...11, 13, ... 90]  # 0 ~ 79 to 1 ~ 90
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}  # 1 ~ 90 to 0 ~ 79
        # int to int
        self.coco_ids_to_class_names = {category['id']: category['name'] for category in self.coco.loadCats(self.coco_ids)}  # len 80

    def __getitem__(self, index):
        if index >= len(self.img_id):
            raise StopIteration

        img_id = self.img_id[index]

        img_coco = self.coco.loadImgs(ids=img_id)[0]
        file_name = img_coco['file_name']
        file_path = os.path.join(self.root, 'images', self.set_name, file_name)

        # eg. '/YDE/COCO/images/val2017/000000289343.jpg'
        image = np.array(Image.open(file_path).convert('RGB'))

        annotation_ids = self.coco.getAnnIds(imgIds=img_id)  # img id 에 해당하는 anno id 를 가져온다.
        annotations = [x for x in self.coco.loadAnns(annotation_ids) if x['image_id'] == img_id]       # anno id 에 해당하는 annotation 을 가져온다.

        boxes = np.array([annotation['bbox'] for annotation in annotations], dtype=np.float32)
        try:
            np.testing.assert_equal(np.all([boxes[:, 2] > 0, boxes[:, 3] > 0]), True)  # check error occurring bbox
        except AssertionError:
            del self.img_id[index]
            return self.__getitem__(index)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annotation['category_id'] for annotation in annotations], dtype=np.int32)
        masks = np.array([self.coco.annToMask(annotation) for annotation in annotations], dtype=np.uint8)

        areas = np.array([annotation['area'] for annotation in annotations], dtype=np.float32)
        iscrowd = np.array([annotation['iscrowd'] for annotation in annotations], dtype=np.uint8)

        self.albumentation_transforms = albumentations.Compose([
            albumentations.pytorch.ToTensorV2()
        ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=["labels"]))

        if self.augmentation is not None:
            for augmentation in self.augmentation:
                self.albumentation_transforms.transforms.insert(0, augmentation)

        augmented = self.albumentation_transforms(image=image, bboxes=boxes, mask=masks, labels=labels)
        image = augmented["image"]
        boxes = augmented["bboxes"]
        labels = augmented["labels"]
        masks = augmented["mask"]

        if self.save_visualization:
            cv_image = image.detach().cpu().numpy()
            cv_image = np.transpose(cv_image, (1, 2, 0))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            cv_image = cv_image.astype(np.uint8).copy()

            cats = self.coco.loadCats(self.coco.getCatIds())
            nms = [cat['name'] for cat in cats]
            insert_idx_list = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82]
            for insert_idx in insert_idx_list:
                nms.insert(insert_idx, "-")
            nms.append("-")

            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = map(int, boxes[i])
                label = labels[i]

                cv_image = cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv_image = cv2.putText(cv_image, nms[label - 1], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite("visualization/" + file_name, cv_image)

            cv_mask = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), np.uint8)
            cv_masks = masks.detach().cpu().numpy()
            cv_masks = np.transpose(cv_masks, (1, 2, 0))
            cv_masks = cv_masks.astype(np.uint8).copy()
            for i in range(cv_masks.shape[2]):
                cv_mask = cv2.bitwise_or(cv_mask, cv_masks[:, :, i] * 255)
            cv2.imwrite("visualization/" + file_name[:-4] + "_masks.jpg", cv_mask)

        image = torch.as_tensor(image / 255., dtype=torch.float32)
        result_annotation = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'areas': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.uint8),
            'img_id': torch.as_tensor(img_id, dtype=torch.int64)
        }

        return image, result_annotation

    def __len__(self):
        return len(self.img_id)