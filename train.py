from __future__ import print_function, division

import os
import time
import shutil
import pickle
import logging
import argparse
import cv2
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from lib.config import cfg
from lib.dataset import FashionAIKPSDataSet
from lib.model import PoseNet, CascadePoseNet, MaskPoseNet, PoseNetNoPaf, load_model
from lib.utils import get_logger, Recorder


class SumL2Loss(gl.loss.Loss):

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(SumL2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gl.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = gl.loss._apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.sum(loss, axis=self._batch_axis, exclude=True)


class SigmoidLoss(gl.loss.Loss):

    def __init__(self, weight=5., batch_axis=0, **kwargs):
        super(SigmoidLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, mask, sample_weight=None):
        label = gl.loss._reshape_like(F, label, pred)
        # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
        loss = F.elemwise_mul(loss, mask)
        loss = gl.loss._apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


def forward_backward_v2(net, criterions, ctx, packet, is_train=True):
    ht_criterion, = criterions
    data, heatmap, paf, heatmap_mask, paf_mask = packet
    # split to gpus
    data = gl.utils.split_and_load(data, ctx)
    heatmap = gl.utils.split_and_load(heatmap, ctx)
    heatmap_mask = gl.utils.split_and_load(heatmap_mask, ctx)
    paf = gl.utils.split_and_load(paf ,ctx)
    paf_mask = gl.utils.split_and_load(paf_mask ,ctx)
    # run
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for data_, heatmap_, paf_, heatmap_mask_, paf_mask_ in zip(data, heatmap, paf, heatmap_mask, paf_mask):
        # forward
        out_ = net(data_)
        losses_ = []
        num_stages = len(out_)
        for i in range(num_stages):
            out_[i][0] = nd.elemwise_mul(out_[i][0], heatmap_mask_)
            out_[i][1] = nd.elemwise_mul(out_[i][1], paf_mask_)
            heatmap_ = nd.elemwise_mul(heatmap_, heatmap_mask_)
            paf_ = nd.elemwise_mul(paf_, paf_mask_)
            losses_.append(ht_criterion(out_[i][0], heatmap_))
            losses_.append(ht_criterion(out_[i][1], paf_))
        losses.append(losses_)
        # backward
        if is_train:
            ag.backward(losses_)
    ag.set_recording(False)
    ag.set_training(False)
    return losses


def forward_backward_v3(net, criterions, ctx, packet, is_train=True):
    ht_criterion, = criterions
    data, ht4, mask4, ht8, mask8, ht16, mask16 = packet
    ht = [ht4, ht8, ht16]
    mask = [mask4, mask8, mask16]
    # split to gpus
    data = gl.utils.split_and_load(data, ctx)
    ht = [gl.utils.split_and_load(x, ctx) for x in ht]
    mask = [gl.utils.split_and_load(x, ctx) for x in mask]
    # run
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for data_, ht4_, mask4_, ht8_, mask8_, ht16_, mask16_ in zip(data, ht[0], mask[0], ht[1], mask[1], ht[2], mask[2]):
        # forward
        global_pred, refine_pred = net(data_)
        # global
        losses_ = [ht_criterion(global_pred[0], ht4_, mask4_),
                   ht_criterion(global_pred[1], ht8_, mask8_),
                   ht_criterion(global_pred[2], ht16_, mask16_),
                   ht_criterion(refine_pred, ht4_, mask4_)]
        losses.append(losses_)
        # backward
        if is_train:
            ag.backward(losses_)
    ag.set_recording(False)
    ag.set_training(False)
    return losses


def forward_backward_v4(net, criterions, ctx, packet, is_train=True):
    ht_criterion, obj_criterion = criterions
    data, ht, ht_mask, obj, obj_mask = packet
    # split to gpus
    data = gl.utils.split_and_load(data, ctx)
    ht = gl.utils.split_and_load(ht, ctx)
    ht_mask = gl.utils.split_and_load(ht_mask, ctx)
    obj = gl.utils.split_and_load(obj, ctx)
    obj_mask = gl.utils.split_and_load(obj_mask, ctx)
    # run
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for data_, ht_, ht_mask_, obj_, obj_mask_ in zip(data, ht, ht_mask, obj, obj_mask):
        # forward
        obj_pred, ht_global_pred, ht_refine_pred = net(data_)
        # global
        losses_ = [obj_criterion(obj_pred, obj_, obj_mask_),
                   ht_criterion(ht_global_pred, ht_, ht_mask_),
                   ht_criterion(ht_refine_pred, ht_, ht_mask_)]
        losses.append(losses_)
        # backward
        if is_train:
            ag.backward(losses_)
    ag.set_recording(False)
    ag.set_training(False)
    return losses


def forward_backward_v5(net, criterions, ctx, packet, is_train=True):
    ht_criterion, = criterions
    data, heatmap, paf, heatmap_mask, paf_mask = packet
    # split to gpus
    data = gl.utils.split_and_load(data, ctx)
    heatmap = gl.utils.split_and_load(heatmap, ctx)
    heatmap_mask = gl.utils.split_and_load(heatmap_mask, ctx)
    # run
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for data_, heatmap_, heatmap_mask_ in zip(data, heatmap, heatmap_mask):
        # forward
        out_ = net(data_)
        losses_ = []
        num_stages = len(out_)
        for i in range(num_stages):
            out_[i] = nd.elemwise_mul(out_[i], heatmap_mask_)
            heatmap_ = nd.elemwise_mul(heatmap_, heatmap_mask_)
            losses_.append(ht_criterion(out_[i], heatmap_))
        losses.append(losses_)
        # backward
        if is_train:
            ag.backward(losses_)
    ag.set_recording(False)
    ag.set_training(False)
    return losses


def reduce_losses(losses):
    n = len(losses)
    m = len(losses[0])
    ret = np.zeros(m)
    for i in range(n):
        for j in range(m):
            ret[j] += losses[i][j].mean().asscalar()
    ret /= n
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--freq', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--steps', type=str, default='1000')
    parser.add_argument('--lr-decay', type=float, default=0.1)
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg16', 'vgg19', 'resnet50'])
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--prefix', type=str, default='default', help='model description')
    parser.add_argument('--version', type=int, default=2, choices=[2, 3, 4, 5], help='model version')
    parser.add_argument('--num-stage', type=int, default=5)
    parser.add_argument('--num-channel', type=int, default=64)
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # hyper parameters
    ctx = [mx.gpu(int(x)) for x in args.gpu.split(',')]
    data_dir = cfg.DATA_DIR
    lr = args.lr
    wd = args.wd
    optim = args.optim
    batch_size = args.batch_size
    epoches = args.epoches
    freq = args.freq
    steps = [int(x) for x in args.steps.split(',')]
    lr_decay = args.lr_decay
    backbone = args.backbone
    prefix = args.prefix
    model_path = None if args.model_path == '' else args.model_path
    num_stage = args.num_stage
    num_channel = args.num_channel
    version = args.version
    if version == 2:
        base_name = 'V2.%s-%s-S%d-C%d-BS%d-%s' % (prefix, backbone, num_stage, num_channel, batch_size, optim)
    elif version == 3:
        base_name = 'V3.%s-%s-C%d-BS%d-%s' % (prefix, backbone, num_channel, batch_size, optim)
    elif version == 4:
        base_name = 'V4.%s-%s-C%d-BS%d-%s' % (prefix, backbone, num_channel, batch_size, optim)
    elif version == 5:
        base_name = 'V5.%s-%s-S%d-C%d-BS%d-%s' % (prefix, backbone, num_stage, num_channel, batch_size, optim)
    else:
        raise RuntimeError('no such version %d'%version)
    logger = get_logger()
    # data
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    traindata = FashionAIKPSDataSet(df_train, version=version, is_train=True)
    testdata = FashionAIKPSDataSet(df_test, version=version, is_train=False)
    trainloader = gl.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=4)
    testloader = gl.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=4)
    epoch_size = len(trainloader)
    # model
    if not model_path:
        num_kps = cfg.NUM_LANDMARK
        num_limb = len(cfg.PAF_LANDMARK_PAIR)
        if version == 2:
            net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=num_stage, channels=num_channel)
            creator, featname, fixed = cfg.BACKBONE_v2[backbone]
        elif version == 3:
            net = CascadePoseNet(num_kps=num_kps, num_channel=num_channel)
            creator, featname, fixed = cfg.BACKBONE_v3[backbone]
        elif version == 4:
            net = MaskPoseNet(num_kps=num_kps, num_channel=num_channel)
            creator, featname, fixed = cfg.BACKBONE_v4[backbone]
        elif version == 5:
            net = PoseNetNoPaf(num_kps=num_kps, stages=num_stage, channels=num_channel)
            creator, featname, fixed = cfg.BACKBONE_v2[backbone]
        else:
            raise RuntimeError('no such version %d'%version)
        net.init_backbone(creator, featname, fixed, pretrained=True)
        net.initialize(mx.init.Normal(), ctx=ctx)
    else:
        logger.info('Load net from %s', model_path)
        net = load_model(model_path, version=version)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    ht_criterion = SumL2Loss()
    obj_criterion = SigmoidLoss()
    ht_criterion.hybridize()
    obj_criterion.hybridize()
    # trainer
    steps = [epoch_size * x for x in steps]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_decay)
    if optim == 'sgd':
        trainer = gl.trainer.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd, 'momentum': 0.9, 'lr_scheduler': lr_scheduler})
    else:
        trainer = gl.trainer.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'lr_scheduler': lr_scheduler})
    # logger
    log_dir = './log-v%d/%s' % (version, base_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    sw = SummaryWriter(log_dir)
    if version == 2:
        rds = []
        for i in range(num_stage):
            rd1 = Recorder('h-%d' % i, freq)
            rd2 = Recorder('p-%d' % i, freq)
            rds.append(rd1)
            rds.append(rd2)
    elif version == 3:
        rds = [Recorder('Global-04', freq), Recorder('Global-08', freq), Recorder('Global-16', freq), Recorder('Refine', freq)]
    elif version == 4:
        rds = [Recorder('mask', freq), Recorder('ht_global', freq), Recorder('ht_refine', freq)]
    elif version == 5:
        rds = [Recorder('h-%d' % i, freq) for i in range(num_stage)]
    else:
        raise RuntimeError('no such version %d'%version)
    # meta info
    global_step = 0
    # forward and backward
    if version == 2:
        forward_backward = forward_backward_v2
        criterions = (ht_criterion,)
    elif version == 3:
        forward_backward = forward_backward_v3
        criterions = (ht_criterion,)
    elif version == 4:
        forward_backward = forward_backward_v4
        criterions = (ht_criterion, obj_criterion)
    elif version == 5:
        forward_backward = forward_backward_v5
        criterions = (ht_criterion,)
    else:
        raise RuntimeError('no such version %d'%version)
    for epoch_idx in range(epoches):
        # train part
        tic = time.time()
        for rd in rds:
            rd.reset()
        sw.add_scalar('lr', trainer.learning_rate, global_step)
        for batch_idx, packet in enumerate(trainloader):
            # [(l1, l2, ...), (l1, l2, ...)]
            losses = forward_backward(net, criterions, ctx, packet, is_train=True)
            trainer.step(batch_size)
            # reduce to [l1, l2, ...]
            ret = reduce_losses(losses)
            for rd, loss in zip(rds, ret):
                rd.update(loss)
            if batch_idx % freq == freq - 1:
                for rd in rds:
                    name, value = rd.get()
                    sw.add_scalar('train/' + name, value, global_step)
                    logger.info('[Epoch %d][Batch %d] %s = %f', epoch_idx + 1, batch_idx + 1, name, value)
                global_step += 1
                toc = time.time()
                speed = (batch_idx + 1) * batch_size / (toc - tic)
                logger.info('[Epoch %d][Batch %d] Speed = %.2f sample/sec', epoch_idx + 1, batch_idx + 1, speed)
        toc = time.time()
        logger.info('[Epoch %d] Global step %d', epoch_idx + 1, global_step - 1)
        logger.info('[Epoch %d] Train Cost %.0f sec', epoch_idx + 1, toc - tic)
        # test part
        tic = time.time()
        for rd in rds:
            rd.reset()
        for batch_idx, packet in enumerate(testloader):
            losses = forward_backward(net, criterions, ctx, packet, is_train=False)
            ret = reduce_losses(losses)
            for rd, loss in zip(rds, ret):
                rd.update(loss)
        for rd in rds:
            name, value = rd.get()
            sw.add_scalar('test/' + name, value, global_step)
            logger.info('[Epoch %d][Test] %s = %f', epoch_idx + 1, name, value)
        toc = time.time()
        logger.info('[Epoch %d] Test Cost %.0f sec', epoch_idx + 1, toc - tic)
        # save part
        save_path = './output/%s-%04d.params' % (base_name, epoch_idx + 1)
        net.save_params(save_path)
        logger.info('[Epoch %d] Saved to %s', epoch_idx + 1, save_path)


if __name__ == '__main__':
    main()
