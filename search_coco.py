import os

import torch
import ray
import time
import numpy as np
import random
import json
import time
import logging

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments

from augmentations import smallobjectaugmentation_list, appendAlbumentation
from archive import remove_deplicates, policy_decoder

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cocoutils import get_dataloaders, get_valid_dataloaders, get_coco_stats
from loggingutil import get_logger, add_filehandler

from models import MaskRCNN

# Set cuda
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

logger = get_logger('SmallObjectAugmentFAA')


@ray.remote(num_cpus=8, num_gpus=1)
def train_model(model_path, num_epochs, cross_valid_fold, num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = MaskRCNN(num_classes=num_classes).to(device)

    # if exist model, evaluate model after load
    if os.path.exists(model_path):
        valid_data_loader = get_valid_dataloaders(dataroot=dataroot, type='val', batch_size=batch_size)

        print("%s Model Exist! Load Model.." % model_path)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        print('----------------------COCOeval Metric start--------------------------')
        inference_results = []
        for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
            with torch.no_grad():
                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                inference = model(imgs)

                for batch_idx in range(len(images_batch)):
                    boxes, labels, scores, mask = inference[batch_idx]["boxes"], inference[batch_idx]["labels"].cpu(), inference[batch_idx]["scores"].cpu(), inference[batch_idx]["masks"].cpu()

                    if len(boxes) > 0:
                        boxes[:, 2] -= boxes[:, 0]
                        boxes[:, 3] -= boxes[:, 1]
                        boxes = boxes.tolist()
                        # boxes = [list(map(round, box)) for box in boxes]

                        for box_id in range(len(boxes)):
                            # if scores[box_id] < 0.5:
                            #     continue

                            box = boxes[box_id]
                            label = labels[box_id]
                            score = scores[box_id]

                            image_result = {
                                'image_id': annotations[batch_idx]["img_id"].cpu().item(),
                                'category_id': label.item(),
                                'bbox': box,
                                'score': score.item(),
                            }

                            inference_results.append(image_result)

        json.dump(inference_results, open("instances_val2017_bbox_" + str(cross_valid_fold) + ".json", 'w'), indent=4)

        coco_gt = COCO(annotation_file="/YDE/COCO/annotations/instances_val2017.json")
        coco_pred = coco_gt.loadRes(resFile="instances_val2017_bbox_" + str(cross_valid_fold) + ".json")

        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        result = get_coco_stats(coco_eval.stats, False)

        print('----------------------COCOeval Metric end--------------------------')
        del valid_data_loader

        return model, cross_valid_fold, result

    else:  # not exist, train model
        print("%s Model not Exist! Train Model.." % model_path)
        print('----------------------train start--------------------------')

        train_data_loader, valid_data_loader = get_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=cross_valid_fold)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        writer_loss = SummaryWriter(log_dir='logs/%d-fold/loss' % cross_valid_fold)
        writer_loss_classifier = SummaryWriter(log_dir='logs/%d-fold/loss_classifier' % cross_valid_fold)
        writer_loss_mask = SummaryWriter(log_dir='logs/%d-fold/loss_mask' % cross_valid_fold)
        writer_loss_box_reg = SummaryWriter(log_dir='logs/%d-fold/loss_box_reg' % cross_valid_fold)
        writer_loss_objectness = SummaryWriter(log_dir='logs/%d-fold/loss_objectness' % cross_valid_fold)

        min_valid_loss = 999999
        for epoch in range(1, num_epochs + 1):
            train_loss, train_loss_classifier, train_loss_mask, train_loss_box_reg, train_loss_objectness = 0, 0, 0, 0, 0
            valid_loss, valid_loss_classifier, valid_loss_mask, valid_loss_box_reg, valid_loss_objectness = 0, 0, 0, 0, 0
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
                train_loss_mask += loss_dict['loss_mask'].item()
                train_loss_box_reg += loss_dict['loss_box_reg'].item()
                train_loss_objectness += loss_dict['loss_objectness'].item()

            train_loss /= len(train_data_loader)
            train_loss_classifier /= len(train_data_loader)
            train_loss_mask /= len(train_data_loader)
            train_loss_box_reg /= len(train_data_loader)
            train_loss_objectness /= len(train_data_loader)

            writer_loss.add_scalar('loss', train_loss, epoch)
            writer_loss_classifier.add_scalar('loss_classifier', train_loss_classifier, epoch)
            writer_loss_mask.add_scalar('loss_mask', train_loss_mask, epoch)
            writer_loss_box_reg.add_scalar('loss_box_reg', train_loss_box_reg, epoch)
            writer_loss_objectness.add_scalar('loss_objectness', train_loss_objectness, epoch)

            print(f"train epoch : {epoch}, cross_valid_fold : {cross_valid_fold}, loss_classifier: {train_loss_classifier:.5f}, loss_mask: {train_loss_mask:.5f}, "
                  f"loss_box_reg: {train_loss_box_reg:.5f}, loss_objectness: {train_loss_objectness:.5f}, Total_loss: {train_loss:.5f}")

            for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
                with torch.no_grad():
                    imgs = list(img.to(device) for img in images_batch)
                    annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                    loss_dict = model(imgs, annotations)

                    losses = sum(loss for loss in loss_dict.values())

                    valid_loss += losses
                    valid_loss_classifier += loss_dict['loss_classifier'].item()
                    valid_loss_mask += loss_dict['loss_mask'].item()
                    valid_loss_box_reg += loss_dict['loss_box_reg'].item()
                    valid_loss_objectness += loss_dict['loss_objectness'].item()

            valid_loss /= len(valid_data_loader)
            valid_loss_classifier /= len(valid_data_loader)
            valid_loss_mask /= len(valid_data_loader)
            valid_loss_box_reg /= len(valid_data_loader)
            valid_loss_objectness /= len(valid_data_loader)

            print(f"valid epoch : {epoch}, cross_valid_fold : {cross_valid_fold}, loss_classifier: {valid_loss_classifier:.5f}, loss_mask: {valid_loss_mask:.5f}, "
                f"loss_box_reg: {valid_loss_box_reg:.5f}, loss_objectness: {valid_loss_objectness:.5f}, Total_loss: {valid_loss:.5f}")

            # Model Save
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'valid_loss': min_valid_loss,
                    'optimizer': optimizer.state_dict,
                    'state_dict': model.state_dict()
                }, model_path)

        writer_loss.close()
        writer_loss_classifier.close()
        writer_loss_mask.close()
        writer_loss_box_reg.close()
        writer_loss_objectness.close()

        print('----------------------train end--------------------------')

        print('----------------------COCOeval Metric start--------------------------')
        model.eval()
        inference_results = []
        for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
            with torch.no_grad():
                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                inference = model(imgs)

                for batch_idx in range(len(images_batch)):
                    boxes, labels, scores, mask = inference[batch_idx]["boxes"], inference[batch_idx]["labels"].cpu(), inference[batch_idx]["scores"].cpu(), inference[batch_idx]["masks"].cpu()

                    if len(boxes) > 0:
                        boxes[:, 2] -= boxes[:, 0]
                        boxes[:, 3] -= boxes[:, 1]
                        boxes = boxes.tolist()
                        # boxes = [list(map(round, box)) for box in boxes]

                        for box_id in range(len(boxes)):
                            box = boxes[box_id]
                            label = labels[box_id]
                            score = scores[box_id]

                            # if score < threshold:
                            #     break

                            image_result = {
                                'image_id': annotations[batch_idx]["img_id"].cpu().item(),
                                'bbox': box,
                                'category_id': label.item(),
                                'score': score.item(),
                            }

                            inference_results.append(image_result)

        json.dump(inference_results, open("instances_val2017_bbox_" + str(cross_valid_fold) + ".json", 'w'), indent=4)

        coco_gt = COCO(annotation_file="/YDE/COCO/annotations/instances_val2017.json")
        coco_pred = coco_gt.loadRes(resFile="instances_val2017_bbox_" + str(cross_valid_fold) + ".json")

        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        result = get_coco_stats(coco_eval.stats, False)

        print('----------------------COCOeval Metric end--------------------------')
        del train_data_loader
        del valid_data_loader

        return model, cross_valid_fold, result


def eval_tta(augment, reporter):
    cross_valid_ratio, cross_valid_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']
    batch_size = augment['batch_size']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = MaskRCNN(num_classes=num_classes).to(device)

    checkpoint = torch.load("/YDE/SmallObjectAugmentFAA/" + save_path)
    model.load_state_dict(checkpoint["state_dict"])
    # model.eval()

    polices = policy_decoder(augment, augment["num_policy"], augment["num_op"])
    valid_loaders = []
    for _ in range(augment["num_policy"]):
        augmentation = []
        policy = random.choice(polices)
        for name, pr, level in policy:
            appendAlbumentation(augmentation, name, pr, level)
        _, valid_data_loader = get_dataloaders(dataroot=dataroot, type='train', batch_size=batch_size, fold_idx=cross_valid_fold, augmentation=augmentation)

        valid_loaders.append(valid_data_loader)

    loss = []
    for valid_loader in valid_loaders:
        for i, (images_batch, annotations_batch) in enumerate(valid_loader):
            with torch.no_grad():
                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                loss_dict = model(imgs, annotations)

                losses = sum(loss for loss in loss_dict.values())

                loss.append(losses.item())
    reporter(loss=np.mean(loss), metric=0, elapsed_time=0)

    return np.mean(loss)


if __name__ == "__main__":
    dataroot = "/YDE/COCO"
    dataset = "COCO"
    model = "Mask_R-CNN"
    # until = 5
    num_op = 2
    num_policy = 5
    num_search = 200
    cross_valid_num = 4
    cross_valid_ratio = 0.25
    num_epochs = 200
    num_classes = 91
    batch_size = 8

    add_filehandler(logger, os.path.join('models', '%s_%s.log' % (dataset, model)))
    logger.info('configuration...')

    logger.info('initialize ray...')
    ray.init(num_cpus=32, num_gpus=4, webui_host='127.0.0.1')
    logger.info("%s" % ray.cluster_resources())

    num_result_per_cv = 10
    k_fold_model_paths = ['models/%s_%s_fold%d.pth' % (model, dataset, i + 1) for i in range(cross_valid_num)]

    logger.info('----- Train without Augmentations, cv=%d ratio=%.2f -----' % (cross_valid_num, cross_valid_ratio))
    parallel_train = [train_model.remote(model_path=k_fold_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes) for i in range(cross_valid_num)]

    tqdm_epoch = tqdm(range(num_epochs))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs_per_cv = OrderedDict()
            for cross_valid_idx in range(cross_valid_num):
                try:
                    latest_ckpt = torch.load(k_fold_model_paths[cross_valid_idx])
                    if 'epoch' not in latest_ckpt:
                        epochs_per_cv['cv%d' % (cross_valid_idx + 1)] = -1
                        continue
                    epochs_per_cv['cv%d' % (cross_valid_idx + 1)] = latest_ckpt['epoch']
                except Exception as e:
                    # print(e)
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == cross_valid_num and min(epochs_per_cv.values()) >= num_epochs:
                is_done = True
            if len(epochs_per_cv) == cross_valid_num and min(epochs_per_cv.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    model_results = ray.get(parallel_train)
    for r_model, r_cv, r_dict in model_results:
        del r_model
        logger.info('model=%s cv=%d AP=%.4f APSmall=%.4f' % (model, r_cv, r_dict['AP_IoU05_95_area_all_maxDets100'], r_dict['AP_IoU05_95_area_small_maxDets100']))

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
    reward_metric = 'loss'
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
        results = sorted(results, key=lambda x: x.last_result[reward_metric])

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
    parallel_train_optimal_augment = [train_model.remote(model_path=k_fold_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes) for i in range(cross_valid_num)] + \
                                     [train_model.remote(model_path=k_fold_model_paths[i], num_epochs=num_epochs, cross_valid_fold=i + 1, num_classes=num_classes) for i in range(cross_valid_num)]

    tqdm_epoch = tqdm(num_epochs)
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(cross_valid_num):
                try:
                    if os.path.exists(k_fold_default_augment_model_paths[exp_idx]):
                        latest_ckpt = torch.load(k_fold_default_augment_model_paths[exp_idx])
                        epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except Exception as e:
                    print(e)
                    pass
                try:
                    if os.path.exists(k_fold_optimal_augment_model_paths[exp_idx]):
                        latest_ckpt = torch.load(k_fold_optimal_augment_model_paths[exp_idx])
                        epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except Exception as e:
                    print(e)
                    pass

            tqdm_epoch.set_postfix(epochs)
            if len(epochs) == cross_valid_num * 2 and min(epochs.values()) >= num_epochs:
                is_done = True
            if len(epochs) == cross_valid_num * 2 and min(epochs.values()) >= epoch:
                break
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
            logger.info('[%s] AP=%.4f APSmall=%.4f' % (train_mode, r_dict['AP_IoU05_95_area_all_maxDets100'], r_dict['AP_IoU05_95_area_small_maxDets100']))
            APavg += r_dict['AP_IoU05_95_area_all_maxDets100']
            APSmallavg += r_dict['AP_IoU05_95_area_small_maxDets100']

        APavg /= cross_valid_num
        APSmallavg /= cross_valid_num
        logger.info('[%s] AP average=%.4f APSmall average=%.4f (#cross_valid_num=%d)' % (train_mode, APavg, APSmallavg, cross_valid_num))
