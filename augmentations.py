# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import albumentations, cv2, os

from albumentations.core.transforms_interface import DualTransform
from torchvision.transforms.transforms import Compose

random_mirror = True

from typing import Tuple, Any, Dict

BoxInternalType = Tuple[float, float, float, float]

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


def albumentation_augment_list():  # 16 oeprations and their ranges
    l = [
        (ShearX, -10, 10),  # 0
        (ShearY, -10, 10),  # 1
        (TranslateX, -0.1, 0.1),  # 2
        (TranslateY, -0.1, 0.1),  # 3
        (Rotate, -15, 15),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
    ]
    return l


def appendTorchvision2Albumentation(augmentation_list, name, pr, level):
    if name == "ShearX":
        augmentation_list.append(albumentations.Affine(shear=-5, p=pr))
    elif name == "ShearY":
        augmentation_list.append(albumentations.Affine(shear=5, p=pr))
    elif name == "TranslateX":
        augmentation_list.append(albumentations.Affine(translate_px=5, p=pr))
    elif name == "TranslateY":
        augmentation_list.append(albumentations.Affine(translate_px=5, p=pr))
    elif name == "Rotate":
        augmentation_list.append(albumentations.Affine(rotate=10, p=pr))
    elif name == "AutoContrast":
        pass
    elif name == "Invert":
        augmentation_list.append(albumentations.InvertImg(p=pr))
    elif name == "Equalize":
        augmentation_list.append(albumentations.Equalize(p=pr))
    elif name == "Solarize":
        augmentation_list.append(albumentations.Solarize(p=pr))
    elif name == "Posterize":
        augmentation_list.append(albumentations.Posterize(p=pr))
    elif name == "Contrast":
        augmentation_list.append(albumentations.RandomContrast(p=pr))
    elif name == "Color":
        augmentation_list.append(albumentations.ColorJitter(p=pr))
    elif name == "Brightness":
        augmentation_list.append(albumentations.RandomBrightness(p=pr))
    elif name == "Sharpness":
        augmentation_list.append(albumentations.Sharpen(p=pr))
    elif name == "Cutout":
        # augmentation_list.append(albumentations.CoarseDropout(max_holes=1, p=pr))
        pass


class SmallObjectAugmentation(DualTransform):
    def __init__(self, thresh=32*32, copy_times=1, find_copy_area_epoch=30, all_objects=False, one_object=True, always_apply=False, p=0.5):
        """
        sample = {'img':img, 'annot':annots}
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh：the detection threshold of the small object. If annot_h * annot_w < thresh, the object is small
        p: the prob to do small object augmentation
        find_copy_area_epoch: the epochs to do find copy area
        """

        super().__init__(always_apply, p)
        self.thresh = thresh
        self.p = p
        self.copy_times = copy_times
        self.find_copy_area_epoch = find_copy_area_epoch
        self.all_objects = all_objects
        self.one_object = one_object

        if self.all_objects or self.one_object:
            self.copy_times = 1

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # print("apply_with_params")
        # print("params", params)
        # print("kwargs", kwargs)
        print(kwargs["mask"].shape)

        return kwargs

    def augment(self, sample):
        if self.all_objects and self.one_object:
            return sample

        if np.random.rand() > self.p:
            return sample

        image, bboxes labels, mask = sample['image'], sample['bboxes'], sample['labels'], sample['mask']
        h, w = image.shape[0], image.shape[1]

        small_object_list = list()
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx]
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if self.issmallobject(bbox_h, bbox_w):
                small_object_list.append(idx)

        l = len(small_object_list)
        # No Small Object
        if l == 0:
            return sample

        # Refine the copy_object by the given policy
        # Policy 2:
        copy_object_num = np.random.randint(0, l)
        # Policy 3:
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1

        random_list = random.sample(range(l), copy_object_num)
        bbox_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_bboxes = bboxes[bbox_idx_of_small_object, :]
        bboxes = bboxes.tolist()
        for idx in range(copy_object_num):
            bbox = select_bboxes[idx]
            box_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            for i in range(self.copy_times):
                new_bbox = self.create_copy_bbox(h, w, bbox, bboxes)
                if new_bbox is not None:
                    image = self.add_patch_in_img(new_bbox, bbox, img)
                    bboxes.append(new_bbox)

        return {'image': image, 'bboxes': np.array(bboxes)}

    # def get_params(self):
    #     # print("factor")
    #     # Random int in the range [0, 3]
    #     return {"factor": random.randint(0, 3)}
    #
    # def get_params_dependent_on_targets(self, params):
    #     print(params)
    #     return {"params": params}

    def issmallobject(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False

    def compute_overlap(self, annot_a, annot_b):
        if annot_a is None:
            return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def donot_overlap(self, new_annot, annots):
        for annot in annots:
            if self.compute_overlap(new_annot, annot):
                return False
        return True

    def create_copy_bbox(self, h, w, annot, annots):
        annot = annot.astype(np.int)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        for epoch in range(self.find_copy_area_epoch):
            random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                 np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]]).astype(np.int)

            if self.donot_overlap(new_annot, annots) is False:
                continue

            return new_annot
        return None

    def add_patch_in_img(self, annot, copy_annot, image):
        copy_annot = copy_annot.astype(np.int)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        return image


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)
