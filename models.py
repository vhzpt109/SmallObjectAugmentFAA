from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet


def MaskRCNN(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def getFasterRCNN(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes, min_size=1024, max_size=1024, rpn_nms_thresh=0.3, box_nms_thresh=0.8)

    return model


def geRetinaNet(num_classes):
    backbone = retinanet_resnet50_fpn(pretrained=True)
    model = RetinaNet(backbone, num_classes=num_classes, min_size=1024, max_size=1024, nms_thresh=0.3)

    return model
