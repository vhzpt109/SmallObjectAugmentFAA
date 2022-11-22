import os
import torch
import cv2
import numpy as np
import albumentations
import albumentations.pytorch
import glob
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from augmentations import SmallObjectAugmentation


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODataset(Dataset):
    def __init__(self,
                 root='/YDE/COCO',
                 split='train',
                 augmentation=None,
                 save_visualization=False):
        super().__init__()

        self.root = root

        assert split in ['train', 'val', 'test']
        self.split = split
        self.set_name = split + '2017'

        self.augmentation = augmentation

        self.save_visualization = save_visualization

        self.img_path = glob.glob(os.path.join(self.root, 'images', self.set_name, '*.jpg'))
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_' + self.set_name + '.json'))

        self.img_id = list(self.coco.imgToAnns.keys())
        # self.ids = self.coco.getImgIds()

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
        # image = Image.open(file_path).convert('RGB')
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        masks = np.transpose(masks, (1, 2, 0))

        areas = np.array([annotation['area'] for annotation in annotations], dtype=np.float32)
        iscrowd = np.array([annotation['iscrowd'] for annotation in annotations], dtype=np.uint8)

        self.albumentation_transforms = albumentations.Compose([
            albumentations.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            albumentations.pytorch.ToTensorV2(transpose_mask=True),
        ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=["labels"], min_area=1))

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

        result_annotation = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'areas': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.uint8),
            'img_id': torch.as_tensor(img_id, dtype=torch.int64)
        }

        return image, result_annotation

    def make_det_annos(self, anno):
        annotations = np.zeros((0, 5))
        for idx, anno_dict in enumerate(anno):

            if anno_dict['bbox'][2] < 1 or anno_dict['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = anno_dict['bbox']

            annotation[0, 4] = self.coco_ids_to_continuous_ids[anno_dict['category_id']]  # 원래 category_id가 18이면 들어가는 값은 16
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def __len__(self):
        return len(self.img_id)