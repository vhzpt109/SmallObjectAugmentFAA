import os
import time
import warnings
from collections import OrderedDict

import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augmentations import ApplyFoundPolicy
from dotautils import get_kfold_dataloaders, get_dataloaders
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

    model = getFasterRCNN(num_classes=num_classes).to(device)

    metrics = MAPMetrics()

    # if exist model, evaluate model after load
    if os.path.exists(model_path):
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

    else:  # not exist, train model
        print("%s Model not Exist! Train Model.." % model_path)
        print('----------------------train start--------------------------')

        if is_final:
            train_data_loader = get_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, augmentation=augmentation)
            valid_data_loader = get_dataloaders(dataroot=dataroot, type='val', batch_size=batch_size)
        else:
            train_data_loader, valid_data_loader = get_kfold_dataloaders(dataroot=dataroot, type='train',
                                                                         batch_size=batch_size,
                                                                         fold_idx=cross_valid_fold,
                                                                         augmentation=augmentation)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        writer_loss = SummaryWriter(log_dir='logs/%d-fold/loss' % cross_valid_fold)
        writer_loss_classifier = SummaryWriter(log_dir='logs/%d-fold/loss_classifier' % cross_valid_fold)
        writer_loss_box_reg = SummaryWriter(log_dir='logs/%d-fold/loss_box_reg' % cross_valid_fold)
        writer_loss_objectness = SummaryWriter(log_dir='logs/%d-fold/loss_objectness' % cross_valid_fold)
        writer_loss_rpn_box_reg = SummaryWriter(log_dir='logs/%d-fold/loss_rpn_box_reg' % cross_valid_fold)
        writer_map = SummaryWriter(log_dir='logs/%d-fold/map' % cross_valid_fold)
        writer_map_small = SummaryWriter(log_dir='logs/%d-fold/map_small' % cross_valid_fold)

        map_max = 0
        best_result = None
        for epoch in range(1, num_epochs + 1):
            train_loss, train_loss_classifier, train_loss_box_reg, train_loss_objectness, train_loss_rpn_box_reg = 0, 0, 0, 0, 0
            model.train()
            for i, (images_batch, annotations_batch) in enumerate(train_data_loader):
                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                loss_dict = model(imgs, annotations)

                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss += losses
                train_loss_classifier += loss_dict['loss_classifier'].item()
                train_loss_box_reg += loss_dict['loss_box_reg'].item()
                train_loss_objectness += loss_dict['loss_objectness'].item()
                train_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

            train_loss /= len(train_data_loader)
            train_loss_classifier /= len(train_data_loader)
            train_loss_box_reg /= len(train_data_loader)
            train_loss_objectness /= len(train_data_loader)
            train_loss_rpn_box_reg /= len(train_data_loader)

            writer_loss.add_scalar('train_loss', train_loss, epoch)
            writer_loss_classifier.add_scalar('train_loss_classifier', train_loss_classifier, epoch)
            writer_loss_box_reg.add_scalar('train_loss_box_reg', train_loss_box_reg, epoch)
            writer_loss_objectness.add_scalar('train_loss_objectness', train_loss_objectness, epoch)
            writer_loss_rpn_box_reg.add_scalar('train_loss_rpn_box_reg', train_loss_rpn_box_reg, epoch)

            print(
                f"train epoch : {epoch}, cross_valid_fold : {cross_valid_fold}, loss_classifier: {train_loss_classifier:.5f}, loss_box_reg: {train_loss_box_reg:.5f}, "
                f"loss_objectness: {train_loss_objectness:.5f}, loss_rpn_box_reg: {train_loss_rpn_box_reg:.5f}, Total_loss: {train_loss:.5f}")

            targets = []
            preds = []
            model.eval()
            metrics.reset()
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

            writer_map.add_scalar('map', result["map"], epoch)
            writer_map_small.add_scalar('map_small', result["map_small"], epoch)

            print(f"valid epoch : %d, cross_valid_fold : %d, map: %f, map_small: %f" % (epoch, cross_valid_fold, result["map"], result["map_small"]))

            # Model Save
            if map_max < result["map"]:
                map_max = result["map"]
                torch.save({
                    'epoch': epoch,
                    'map': map_max,
                    'optimizer': optimizer.state_dict,
                    'state_dict': model.state_dict()
                }, model_path)
                best_result = result

            epoch_log = open(model_path[:-4] + "_epoch.txt", "w")
            epoch_log.write(str(epoch))
            epoch_log.close()

        writer_loss.close()
        writer_loss_classifier.close()
        writer_loss_box_reg.close()
        writer_loss_objectness.close()
        writer_loss_rpn_box_reg.close()
        writer_map.close()
        writer_map_small.close()

        print('----------------------train end--------------------------')

        del train_data_loader
        del valid_data_loader

        return model, cross_valid_fold, best_result


if __name__ == "__main__":
    dataroot = "/YDE/DOTA/split_ss_dota"
    dataset = "DOTA"
    model = "Faster_R-CNN"
    cross_valid_num = 2
    cross_valid_ratio = 0.25
    num_epochs = 200
    num_classes = 19
    batch_size = 10

    add_filehandler(logger, os.path.join('models', '%s_%s_train_with_finalpolicy.log' % (dataset, model)))
    logger.info('configuration...')

    logger.info('initialize ray...')
    ray.init(num_cpus=32, num_gpus=4, webui_host='127.0.0.1')
    logger.info("%s" % ray.cluster_resources())

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

    k_fold_default_augment_model_paths = ['models/%s_default_augment_fold%d.pth' % (model, i + 1) for i in range(cross_valid_num)]
    k_fold_optimal_augment_model_paths = ['models/%s_optimal_augment_fold%d.pth' % (model, i + 3) for i in range(cross_valid_num)]
    parallel_train_optimal_augment = [train_model.remote(model_path=k_fold_default_augment_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes, augmentation=None, is_final=True) for i in range(cross_valid_num)] + \
                                     [train_model.remote(model_path=k_fold_optimal_augment_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 3, num_classes=num_classes, augmentation=[ApplyFoundPolicy(policies=final_policy_set)], is_final=True) for i in range(cross_valid_num)]

    tqdm_epoch = tqdm(range(num_epochs), leave=True)
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(cross_valid_num):
                try:
                    epoch_log = open(k_fold_default_augment_model_paths[exp_idx][:-4] + "_epoch.txt", "r")
                    epoch_log_value = int(epoch_log.readline().rstrip())
                    epochs['default_exp%d' % (exp_idx + 1)] = epoch_log_value
                except Exception as e:
                    # print(e)
                    pass

                try:
                    epoch_log = open(k_fold_optimal_augment_model_paths[exp_idx][:-4] + "_epoch.txt", "r")
                    epoch_log_value = int(epoch_log.readline().rstrip())
                    epochs['optimal_exp%d' % (exp_idx + 1)] = epoch_log_value
                except Exception as e:
                    # print(e)
                    pass

            tqdm_epoch.set_postfix(epochs)
            if len(epochs) == cross_valid_num and min(epochs.values()) >= num_epochs:
                is_done = True
            if len(epochs) == cross_valid_num and min(epochs.values()) >= epoch:
                break
            if len(epochs) == cross_valid_num and min(epochs.values()) - 2 > epoch:
                pass
            else:
                time.sleep(10)
        if is_done:
            break

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
