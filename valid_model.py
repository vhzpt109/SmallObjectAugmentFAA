import os
import warnings

import ray
import torch

from dotautils import get_kfold_dataloaders#, get_dataloaders
from cocoutils import get_dataloaders
from loggingutil import get_logger, add_filehandler
from metric import MAPMetrics
from models import getFasterRCNN

# Set ignore warnings
warnings.filterwarnings('ignore')

# Set cuda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

logger = get_logger('SmallObjectAugmentFAA')


@ray.remote(num_cpus=8, num_gpus=1)
def train_model(model_path, num_epochs, cross_valid_fold, num_classes, augmentation, is_final):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = MaskRCNN(num_classes=num_classes).to(device)

    metrics = MAPMetrics()

    # if exist model, evaluate model after load
    if is_final:
        valid_data_loader = get_dataloaders(dataroot=dataroot, type='val', batch_size=batch_size)
    else:
        _, valid_data_loader = get_kfold_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size,
                                                     fold_idx=cross_valid_fold, augmentation=augmentation)

    print("%s Model Exist! Load Model.." % model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    targets = []
    preds = []
    for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
        with torch.no_grad():
            imgs = list(img.to(device) for img in images_batch)
            annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

            inference = model(imgs)

            for j in range(len(annotations)):
                boxes_target = annotations[j]["boxes"].cpu()
                boxes_preds = inference[j]["boxes"].cpu()
                labels_target = annotations[j]["labels"].cpu()
                labels_preds = inference[j]["labels"].cpu()
                scores_preds = inference[j]["scores"].cpu()

                targets.append(
                    dict(
                        boxes=boxes_target,
                        labels=labels_target
                    )
                )
                preds.append(
                    dict(
                        boxes=boxes_preds,
                        labels=labels_preds,
                        scores=scores_preds
                    )
                )
    metrics.update(preds=preds, target=targets)
    result = metrics.compute()

    print(f"Model Evaluate Result, cross_valid_fold : %d, map: %f, map_small: %f" % (cross_valid_fold, result["map"], result["map_small"]))

    del valid_data_loader

    return model, cross_valid_fold, result


if __name__ == "__main__":
    dataroot = "/YDE/COCO"
    dataset = "DOTA"
    model = "Faster_R-CNN"
    # until = 5
    num_op = 1
    num_policy = 2
    num_search = 50
    cross_valid_num = 2
    cross_valid_ratio = 0.25
    num_epochs = 100
    num_classes = 19
    batch_size = 12

    add_filehandler(logger, os.path.join('models', '%s_%s.log' % (dataset, model)))
    logger.info('configuration...')

    logger.info('initialize ray...')
    ray.init(num_cpus=32, num_gpus=4, webui_host='127.0.0.1')
    logger.info("%s" % ray.cluster_resources())

    k_fold_default_augment_model_paths = ['models/%s_default_augment_fold%d.pth' % (model, i + 1) for i in range(cross_valid_num)]
    k_fold_optimal_augment_model_paths = ['models/%s_optimal_augment_fold%d.pth' % (model, i + 3) for i in range(cross_valid_num)]
    parallel_train_optimal_augment = [train_model.remote(model_path=k_fold_default_augment_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes, augmentation=None, is_final=True) for i in range(cross_valid_num)] + \
                                     [train_model.remote(model_path=k_fold_optimal_augment_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes, augmentation=None, is_final=True) for i in range(cross_valid_num)]

    logger.info('getting results...')
    final_results = ray.get(parallel_train_optimal_augment)

    # getting final optimal performance
    for train_mode in ['default', 'augment']:
        APavg = 0.
        APSmallavg = 0.
        for _ in range(cross_valid_num):
            _, r_cv, r_dict = final_results.pop(0)
            logger.info('[%s] AP=%.4f APSmall=%.4f' % (train_mode, r_dict['map'], r_dict['map_small']))
            APavg += r_dict['map']
            APSmallavg += r_dict['map_small']

        APavg /= cross_valid_num
        APSmallavg /= cross_valid_num
        logger.info('[%s] AP average=%.4f APSmall average=%.4f (#cross_valid_num=%d)' % (train_mode, APavg, APSmallavg, cross_valid_num))
