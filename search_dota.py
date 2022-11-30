import os
import warnings

import torch
import ray
import time
import numpy as np
import random
import json
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments

from augmentations import smallobjectaugmentation_list, appendAlbumentation
from archive import remove_deplicates, policy_decoder

from dotautils import get_dataloaders, get_valid_dataloaders
from loggingutil import get_logger, add_filehandler

from models import getFasterRCNN

from metric import MAPMetrics

# Set ignore warnings
warnings.filterwarnings('ignore' )

# Set cuda
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

logger = get_logger('SmallObjectAugmentFAA')


@ray.remote(num_cpus=8, num_gpus=1)
def train_model(model_path, num_epochs, cross_valid_fold, num_classes, augmentation):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = getFasterRCNN(num_classes=num_classes).to(device)

    metrics = MAPMetrics()

    # if exist model, evaluate model after load
    if os.path.exists(model_path):
        valid_data_loader = get_valid_dataloaders(dataroot=dataroot, type='val', batch_size=batch_size)

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

                for i in range(len(annotations)):
                    boxes_target = annotations[i]["boxes"].cpu()
                    boxes_preds = inference[i]["boxes"].cpu()
                    labels_target = annotations[i]["labels"].cpu()
                    labels_preds = inference[i]["labels"].cpu()
                    scores_preds = inference[i]["scores"].cpu()

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

        train_data_loader, valid_data_loader = get_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=cross_valid_fold, augmentation=augmentation)

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
            for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
                with torch.no_grad():
                    imgs = list(img.to(device) for img in images_batch)
                    annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                    inference = model(imgs)

                    for i in range(len(annotations)):
                        boxes_target = annotations[i]["boxes"].cpu()
                        boxes_preds = inference[i]["boxes"].cpu()
                        labels_target = annotations[i]["labels"].cpu()
                        labels_preds = inference[i]["labels"].cpu()
                        scores_preds = inference[i]["scores"].cpu()

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


def eval_tta(augment, reporter):
    cross_valid_ratio, cross_valid_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']
    batch_size = augment['batch_size']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = getFasterRCNN(num_classes=num_classes).to(device)

    metrics = MAPMetrics()

    checkpoint = torch.load("/YDE/SmallObjectAugmentFAA/" + save_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    polices = policy_decoder(augment, augment["num_policy"], augment["num_op"])
    valid_loaders = []
    for _ in range(augment["num_policy"]):
        augmentation = []
        policy = random.choice(polices)
        for name, pr, level in policy:
            appendAlbumentation(augmentation, name, pr, level)
        _, valid_data_loader = get_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=cross_valid_fold, augmentation=augmentation)

        valid_loaders.append(valid_data_loader)

    maps = []
    maps_small = []
    for valid_loader in valid_loaders:
        targets = []
        preds = []
        for i, (images_batch, annotations_batch) in enumerate(valid_loader):
            with torch.no_grad():
                model.eval()
                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                inference = model(imgs)

                for i in range(len(annotations)):
                    boxes_target = annotations[i]["boxes"].cpu()
                    boxes_preds = inference[i]["boxes"].cpu()
                    labels_target = annotations[i]["labels"].cpu()
                    labels_preds = inference[i]["labels"].cpu()
                    scores_preds = inference[i]["scores"].cpu()

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
        print(result)

        maps.append(result["map"])
        maps_small.append(result["map_small"])

    reporter(map=np.mean(maps), map_small=np.mean(maps_small), elapsed_time=0)

    return np.mean(maps)


if __name__ == "__main__":
    dataroot = "/YDE/DOTA/split_ss_dota"
    dataset = "DOTA"
    model = "Faster_R-CNN"
    # until = 5
    num_op = 2
    num_policy = 5
    num_search = 200
    cross_valid_num = 4
    cross_valid_ratio = 0.25
    num_epochs = 200
    num_classes = 18
    batch_size = 12

    add_filehandler(logger, os.path.join('models', '%s_%s.log' % (dataset, model)))
    logger.info('configuration...')

    logger.info('initialize ray...')
    ray.init(num_cpus=32, num_gpus=4, webui_host='127.0.0.1')
    logger.info("%s" % ray.cluster_resources())

    num_result_per_cv = 10
    k_fold_model_paths = ['models/%s_%s_fold%d.pth' % (model, dataset, i + 1) for i in range(cross_valid_num)]

    logger.info('----- Train without Augmentations, cv=%d ratio=%.2f -----' % (cross_valid_num, cross_valid_ratio))
    parallel_train = [train_model.remote(model_path=k_fold_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes, augmentation=None) for i in range(cross_valid_num)]

    tqdm_epoch = tqdm(range(num_epochs), leave=True)
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs_per_cv = OrderedDict()
            for cross_valid_idx in range(cross_valid_num):
                try:
                    epoch_log = open(k_fold_model_paths[cross_valid_idx][:-4] + "_epoch.txt", "r")
                    epoch_log_value = int(epoch_log.readline().rstrip())
                    epochs_per_cv['cv%d' % (cross_valid_idx + 1)] = epoch_log_value
                except Exception as e:
                    # print(e)
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == cross_valid_num and min(epochs_per_cv.values()) >= num_epochs:
                is_done = True
            if len(epochs_per_cv) == cross_valid_num and min(epochs_per_cv.values()) >= epoch:
                break
            if len(epochs_per_cv) == cross_valid_num and min(epochs_per_cv.values()) - 2 > epoch:
                pass
            else:
                time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    model_results = ray.get(parallel_train)
    for r_model, r_cv, r_dict in model_results:
        del r_model
        logger.info('model=%s cv=%d AP=%.4f APSmall=%.4f' % (model, r_cv, r_dict['map'], r_dict['map_small']))

    logger.info('----- Search Augmentation Policies -----')
    ops = smallobjectaugmentation_list()
    space = {}
    for i in range(num_policy):
        for j in range(num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)

    final_policy_set = []
    total_computation = 0
    reward_metric = 'map'
    for cross_valid_fold in range(1, cross_valid_num + 1):
        name = "search_%s_%s_fold%d_ratio%.2f" % (dataset, model, cross_valid_fold, cross_valid_ratio)
        print(name)
        register_trainable(name, lambda augment, reporter: eval_tta(augment=augment, reporter=reporter))
        algo = HyperOptSearch(space, max_concurrent=4 * 20, metric=reward_metric)

        exp_config = {
            name: {
                'run': name,
                'num_samples': num_search,
                'resources_per_trial': {'cpu': 8, 'gpu': 1},
                'stop': {'training_iteration': num_policy},
                'config': {
                    'dataroot': dataroot, 'save_path': k_fold_model_paths[cross_valid_fold - 1],
                    'batch_size': batch_size,
                    'cv_ratio_test': cross_valid_ratio, 'cv_fold': cross_valid_fold,
                    'num_op': num_op, 'num_policy': num_policy
                },
            }
        }
        results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True,
                                  resume=False, raise_on_failed_trial=True)
        results = [x for x in results if x.last_result is not None]
        results = sorted(results, key=lambda x: x.last_result[reward_metric], reverse=True)

        # calculate computation usage
        for result in results:
            total_computation += result.last_result['elapsed_time']

        for result in results[:num_result_per_cv]:
            final_policy = policy_decoder(result.config, num_policy, num_op)
            logger.info('loss=%.12f %s' % (result.last_result['loss'], final_policy))

            final_policy = remove_deplicates(final_policy)
            final_policy_set.extend(final_policy)

    logger.info(json.dumps(final_policy_set))
    logger.info('final_policy=%d' % len(final_policy_set))
    # logger.info('processed in %.4f secs, gpu hours=%.4f' % (w.pause('search'), total_computation / 3600.))
    logger.info('----- Train with Augmentations, model=%s dataset=%s aug=%s ratio(=%.2f -----' % (model, dataset, "", cross_valid_ratio))

    k_fold_default_augment_model_paths = ['%s_default_augment_fold%d' % (model, i) for i in range(cross_valid_num)]
    k_fold_optimal_augment_model_paths = ['%s_optimal_augment_fold%d' % (model, i) for i in range(cross_valid_num)]
    parallel_train_optimal_augment = [train_model.remote(model_path=k_fold_default_augment_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes, augmentation=None) for i in range(cross_valid_num)] + \
                                     [train_model.remote(model_path=k_fold_optimal_augment_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes, augmentation=None) for i in range(cross_valid_num)]

    tqdm_epoch = tqdm(num_epochs)
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(cross_valid_num):
                try:
                    epoch_log = open(k_fold_default_augment_model_paths[exp_idx] + "_epoch.txt", "r")
                    epoch_log_value = int(epoch_log.readline().rstrip())
                    epochs['default_exp%d' % (exp_idx + 1)] = epoch_log_value
                except Exception as e:
                    # print(e)
                    pass

                try:
                    epoch_log = open(k_fold_optimal_augment_model_paths[exp_idx] + "_epoch.txt", "r")
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
