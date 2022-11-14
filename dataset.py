import os
import torch
import numpy as np
import albumentations
import glob
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODataset(Dataset):
    def __init__(self,
                 root='/YDE/COCO',
                 split='train',
                 transform=None,
                 visualization=False):
        super().__init__()

        self.root = root

        assert split in ['train', 'val', 'test']
        self.split = split
        self.set_name = split + '2017'

        self.img_path = glob.glob(os.path.join(self.root, 'images', self.set_name, '*.jpg'))
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_' + self.set_name + '.json'))

        self.img_id = list(self.coco.imgToAnns.keys())
        # self.ids = self.coco.getImgIds()

        self.coco_ids = sorted(self.coco.getCatIds())  # list of coco labels [1, ...11, 13, ... 90]  # 0 ~ 79 to 1 ~ 90
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}  # 1 ~ 90 to 0 ~ 79
        # int to int
        self.coco_ids_to_class_names = {category['id']: category['name'] for category in
                                        self.coco.loadCats(self.coco_ids)}  # len 80
        # int to string
        # {1 : 'person', 2: 'bicycle', ...}
        '''
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        '''

    def __getitem__(self, index):
        img_id = self.img_id[index]

        img_coco = self.coco.loadImgs(ids=img_id)[0]
        file_name = img_coco['file_name']
        file_path = os.path.join(self.root, 'images', self.set_name, file_name)

        # eg. '/YDE/COCO/images/val2017/000000289343.jpg'
        image = Image.open(file_path).convert('RGB')

        annotation_ids = self.coco.getAnnIds(imgIds=img_id)  # img id 에 해당하는 anno id 를 가져온다.
        annotations = [x for x in self.coco.loadAnns(annotation_ids) if x['image_id'] == img_id]       # anno id 에 해당하는 annotation 을 가져온다.

        boxes = np.array([annotation['bbox'] for annotation in annotations], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annotation['category_id'] for annotation in annotations], dtype=np.int32)
        masks = np.array([self.coco.annToMask(annotation) for annotation in annotations], dtype=np.uint8)

        area = np.array([annotation['area'] for annotation in annotations], dtype=np.float32)
        iscrowd = np.array([annotation['iscrowd'] for annotation in annotations], dtype=np.uint8)

        result_annotation = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd}

        self.transforms = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        image = self.transforms(image)

        result_annotation['boxes'] = torch.as_tensor(result_annotation['boxes'], dtype=torch.float32)
        result_annotation['masks'] = torch.as_tensor(result_annotation['masks'], dtype=torch.uint8)
        result_annotation['labels'] = torch.as_tensor(result_annotation['labels'], dtype=torch.int64)
        result_annotation['area'] = torch.as_tensor(result_annotation['area'], dtype=torch.float32)
        result_annotation['iscrowd'] = torch.as_tensor(result_annotation['iscrowd'], dtype=torch.uint8)

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