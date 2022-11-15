import os

import torch
import ray
import time
import numpy as np
import torch.distributed as dist
import albumentations
import random

from tqdm import tqdm
from collections import OrderedDict
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments

from augmentations import augment_list, appendTorchvision2Albumentation
from archive import remove_deplicates, policy_decoder

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import COCODataset, collate_fn
from pycocotools.cocoeval import COCOeval

# Set cuda
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


@ray.remote(num_gpus=2, max_calls=1)
def train_model(train_data_loader, valid_data_loader, num_epochs, cross_valid_fold, num_classes, model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = get_model_instance_segmentation(num_classes=num_classes).to(device)

    # if exist model, evaluate model after load
    if os.path.exists(model_path):
        print("%s Model Exist! Load Model.." % model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        # model.eval()

        valid_loss = 0
        for imgs, annotations in valid_data_loader:
            with torch.no_grad():
                break
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations]

                loss_dict = model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                valid_loss += losses
                print(f"Model : {model_path}, loss_classifier: {loss_dict['loss_classifier'].item():.5f}, loss_mask: {loss_dict['loss_mask'].item():.5f}, "
                f"loss_box_reg: {loss_dict['loss_box_reg'].item():.5f}, loss_objectness: {loss_dict['loss_objectness'].item():.5f}, Total_loss: {losses.item():.5f}")

                break

        result = {}
        return model, cross_valid_fold, result
    else:  # not exist, train model
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        print("%s Model not Exist! Train Model.." % model_path)
        print('----------------------train start--------------------------')
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            valid_loss = 0
            for i, (images_batch, annotations_batch) in enumerate(train_data_loader):
                optimizer.zero_grad()

                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]
                # print(imgs)

                loss_dict = model(imgs, annotations)
                # print(loss_dict)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                optimizer.step()

                train_loss += losses
                print(f"train epoch : {epoch + 1}, batch : {i + 1}, loss_classifier: {loss_dict['loss_classifier'].item():.5f}, loss_mask: {loss_dict['loss_mask'].item():.5f}, "
                      f"loss_box_reg: {loss_dict['loss_box_reg'].item():.5f}, loss_objectness: {loss_dict['loss_objectness'].item():.5f}, Total_loss: {losses.item():.5f}")

                torch.save({
                    'epoch': epoch,
                    'log'  : "test",
                    'optimizer': optimizer.state_dict,
                    'state_dict': model.state_dict()
                }, model_path)
                # torch.save(model.state_dict(), model_path)
                break

            model.eval()
            for i, (images_batch, annotations_batch) in enumerate(valid_data_loader):
                with torch.no_grad():
                    imgs = list(img.to(device) for img in images_batch)
                    annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                    loss_dict = model(imgs, annotations)
                    # loss_dict = model(imgs)

                    # a = COCOeval(cocoGt=annotations, cocoDt=loss_dict, iouType="bbox")
                    # a.evaluate()
                    # a.accumulate()
                    # a.summarize()
                    # losses = sum(loss for loss in loss_dict.values())

                    # valid_loss += losses
                    # print(f"valid epoch : {epoch + 1}, batch : {i + 1}, loss_classifier: {loss_dict['loss_classifier'].item():.5f}, loss_mask: {loss_dict['loss_mask'].item():.5f}, "
                    #       f"loss_box_reg: {loss_dict['loss_box_reg'].item():.5f}, loss_objectness: {loss_dict['loss_objectness'].item():.5f}, Total_loss: {losses.item():.5f}")
                    break

                    # Model Save
                    # if epoch % 5 == 0:
                    #     torch.save(model, model_path)
                        # torch.save({
                        #     'epoch': epoch,
                        #     'model_state_dict': model.state_dict()
                        # }, model_path)
        print('----------------------train end--------------------------')

        result = {}

        return model, cross_valid_fold, result


def eval_tta(augment, reporter):
    cross_valid_ratio_test, cross_valid_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # print('Device:', device)
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Count of using GPUs:', torch.cuda.device_count())

    model = get_model_instance_segmentation(num_classes=num_classes).to(device)

    checkpoint = torch.load("/YDE/SmallObjectAugmentFAA/" + save_path)
    model.load_state_dict(checkpoint["state_dict"])
    # model.eval()

    polices = policy_decoder(augment, augment["num_policy"], augment["num_op"])
    valid_loaders = []
    for _ in range(augment["num_policy"]):
        augmentation = []
        policy = random.choice(polices)
        for name, pr, level in policy:
            appendTorchvision2Albumentation(augmentation, name, pr, level)

        # print(augmentation)
        valid_dataset = COCODataset(root=augment["dataroot"], split='val', augmentation=augmentation)

        valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=2,
                                                        collate_fn=collate_fn)
        valid_loaders.append(iter(valid_data_loader))

    loss = []
    for valid_loader in valid_loaders:
        for i, (images_batch, annotations_batch) in enumerate(valid_loader):
            with torch.no_grad():
                imgs = list(img.to(device) for img in images_batch)
                annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations_batch]

                loss_dict = model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())
                loss.append(losses.item())
                break

    reporter(loss=np.mean(loss), metric=0, elapsed_time=0)

    return 1, np.mean(loss)


if __name__ == "__main__":
    dataroot = "/YDE/COCO"
    dataset = "COCO"
    model = "Faster_R-CNN"
    # until = 5
    num_op = 2
    num_policy = 5
    num_search = 50
    cross_valid_num = 5
    cross_valid_ratio = 0.4
    num_epochs = 1
    num_classes = 91
    train_batch_size = 4
    valid_batch_size = 4

    # init ray
    ray.init(num_gpus=2, webui_host='127.0.0.1')
    print(ray.cluster_resources())

    # train_dataset = COCODataset(root=dataroot, split='train', augmentation=None)
    valid_dataset = COCODataset(root=dataroot, split='val', augmentation=None, save_visualization=False)

    train_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=train_batch_size,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    collate_fn=collate_fn)

    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=valid_batch_size,
                                                    shuffle=False,
                                                    num_workers=2,
                                                    collate_fn=collate_fn)

    num_result_per_cv = 10
    k_fold_model_paths = ['models/%s_%s_fold%d.pth' % (model, dataset, i) for i in range(cross_valid_num)]

    parallel_train = [train_model.remote(train_data_loader=train_data_loader, valid_data_loader=valid_data_loader, num_epochs=num_epochs, cross_valid_fold=i, num_classes=num_classes, model_path=k_fold_model_paths[i])
                      for i in range(cross_valid_num)]

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

    # ----- getting train results -----
    model_results = ray.get(parallel_train)
    # for r_model, r_cv, r_dict in model_results:
    #     print('model=%s cross_valid_fold=%d top1_train=%.4f top1_valid=%.4f' % (
    #     r_model, r_cv + 1, r_dict['top1_train'], r_dict['top1_valid']))
    #

    # ------ Search Augmentation Policies -----
    ops = augment_list(False)
    space = {}
    for i in range(num_policy):
        for j in range(num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)

    final_policy_set = []
    total_computation = 0
    reward_metric = 'loss'
    for cross_valid_fold in range(cross_valid_num):
        name = "search_%s_%s_fold%d_ratio%.1f" % (dataset, model, cross_valid_fold, cross_valid_ratio)
        print(name)
        register_trainable(name, lambda augment, reporter: eval_tta(augment=augment, reporter=reporter))
        algo = HyperOptSearch(space, max_concurrent=4 * 20, metric=reward_metric)

        exp_config = {
            name: {
                'run': name,
                'num_samples': num_search,
                'resources_per_trial': {'cpu': 16, 'gpu': 2},
                'stop': {'training_iteration': num_policy},
                'config': {
                    'dataroot': dataroot, 'save_path': k_fold_model_paths[cross_valid_fold],
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
            print('loss=%.12f %s' % (result.last_result['loss'], final_policy))

            final_policy = remove_deplicates(final_policy)
            final_policy_set.extend(final_policy)

    #
    #
    # num_experiments = 5
    # # k_fold_default_augment_model_paths = [_get_path(C.get()['dataset'], model, 'ratio%.1f_default%d' % (cross_valid_ratio, _)) for _ in range(num_experiments)]
    # k_fold_default_augment_model_paths = ['%s_default_augment_fold%d' % (model, i) for i in range(num_experiments)]
    # # k_fold_optimal_augment_model_paths = [_get_path(C.get()['dataset'], model, 'ratio%.1f_augment%d' % (cross_valid_ratio, _)) for _ in range(num_experiments)]
    # k_fold_optimal_augment_model_paths = ['%s_optimal_augment_fold%d' % (model, i) for i in range(num_experiments)]
    # parallel_train_optimal_augment = [train_model.remote() for _ in range(num_experiments)] + [train_model.remote() for
    #                                                                                            _ in
    #                                                                                            range(num_experiments)]
    #
    # tqdm_epoch = tqdm(num_epochs)
    # is_done = False
    # for epoch in tqdm_epoch:
    #     while True:
    #         epochs = OrderedDict()
    #         for exp_idx in range(num_experiments):
    #             try:
    #                 if os.path.exists(k_fold_default_augment_model_paths[exp_idx]):
    #                     latest_ckpt = torch.load(k_fold_default_augment_model_paths[exp_idx])
    #                     epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
    #             except Exception as e:
    #                 print(e)
    #                 pass
    #             try:
    #                 if os.path.exists(k_fold_optimal_augment_model_paths[exp_idx]):
    #                     latest_ckpt = torch.load(k_fold_optimal_augment_model_paths[exp_idx])
    #                     epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
    #             except Exception as e:
    #                 print(e)
    #                 pass
    #
    #         tqdm_epoch.set_postfix(epochs)
    #         if len(epochs) == num_experiments * 2 and min(epochs.values()) >= num_epochs:
    #             is_done = True
    #         if len(epochs) == num_experiments * 2 and min(epochs.values()) >= epoch:
    #             break
    #         time.sleep(10)
    #     if is_done:
    #         break
    #
    # # getting train results
    # final_results = ray.get(parallel_train_optimal_augment)
    #
    # # getting final optimal performance
    # for train_mode in ['default', 'augment']:
    #     avg = 0.
    #     for _ in range(num_experiments):
    #         r_model, r_cv, r_dict = final_results.pop(0)
    #         print('[%s] top1_train=%.4f top1_test=%.4f' % (train_mode, r_dict['top1_train'], r_dict['top1_test']))
    #         avg += r_dict['top1_test']
    #     avg /= num_experiments
    #     print('[%s] top1_test average=%.4f (#experiments=%d)' % (train_mode, avg, num_experiments))
