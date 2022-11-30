from torchmetrics.detection.mean_ap import MeanAveragePrecision


def MAPMetrics():
    return MeanAveragePrecision(iou_type="bbox")