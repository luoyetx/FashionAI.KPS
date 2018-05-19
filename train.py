from __future__ import print_function, division

import os
import time
import shutil
import pickle
import logging
import argparse
import datetime
import cv2
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from lib.config import cfg
from lib.dataset import FashionAIKPSDataSet
from lib.model import PoseNet, CascadePoseNet, load_model
from lib.utils import get_logger, Recorder


class SumL2Loss(gl.loss.Loss):

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(SumL2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, mask):
        pred = F.broadcast_mul(pred, mask)
        label = F.broadcast_mul(label, mask)
        label = gl.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = gl.loss._apply_weighting(F, loss, self._weight/2, None)
        return F.sum(loss, axis=self._batch_axis, exclude=True)


def forward_backward_v2(net, criterion, ctx, packet, is_train=True):
    data, ht8, ht8_mask, paf8, paf8_mask = packet
    # split to gpus
    data = gl.utils.split_and_load(data, ctx)
    ht8 = gl.utils.split_and_load(ht8, ctx)
    ht8_mask = gl.utils.split_and_load(ht8_mask, ctx)
    paf8 = gl.utils.split_and_load(paf8 ,ctx)
    paf8_mask = gl.utils.split_and_load(paf8_mask ,ctx)
    # run
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for data_, ht8_, paf8_, ht8_mask_, paf8_mask_ in zip(data, ht8, paf8, ht8_mask, paf8_mask):
        # forward
        out_ = net(data_)
        losses_ = []
        num_stage = len(out_)
        for i in range(num_stage):
            losses_.append(criterion(out_[i][0], ht8_, ht8_mask_))
            losses_.append(criterion(out_[i][1], paf8_, paf8_mask_))
        losses.append(losses_)
        # backward
        if is_train:
            ag.backward(losses_)
    ag.set_recording(False)
    ag.set_training(False)
    return losses


def forward_backward_v3(net, criterion, ctx, packet, is_train=True):
    data, ht4, ht4_mask, paf4, paf4_mask, ht8, ht8_mask, paf8, paf8_mask, ht16, ht16_mask, paf16, paf16_mask = packet
    ht = [ht4, ht8, ht16]
    paf = [paf4, paf8, paf16]
    ht_mask = [ht4_mask, ht8_mask, ht16_mask]
    paf_mask = [paf4_mask, paf8_mask, paf16_mask]
    # split to gpus
    data = gl.utils.split_and_load(data, ctx)
    ht = [gl.utils.split_and_load(x, ctx) for x in ht]
    paf = [gl.utils.split_and_load(x, ctx) for x in paf]
    ht_mask = [gl.utils.split_and_load(x, ctx) for x in ht_mask]
    paf_mask = [gl.utils.split_and_load(x, ctx) for x in paf_mask]
    # run
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for idx, data_ in enumerate(data):
        # forward
        g_ht4, g_paf4, r_ht4, r_paf4, g_ht8, g_paf8, r_ht8, r_paf8, g_ht16, g_paf16, r_ht16, r_paf16 = net(data_)
        ht4_, ht8_, ht16_ = [h[idx] for h in ht]
        paf4_, paf8_, paf16_ = [p[idx] for p in paf]
        ht4_mask_, ht8_mask_, ht16_mask_ = [hm[idx] for hm in ht_mask]
        paf4_mask_, paf8_mask_, paf16_mask_ = [pm[idx] for pm in paf_mask]
        # loss
        losses_ = [criterion(g_ht4, ht4_, ht4_mask_),
                   criterion(r_ht4, ht4_, ht4_mask_),
                   criterion(g_ht8, ht8_, ht8_mask_),
                   criterion(r_ht8, ht8_, ht8_mask_),
                   criterion(g_ht16, ht16_, ht16_mask_),
                   criterion(r_ht16, ht16_, ht16_mask_),
                   criterion(g_paf4, paf4_, paf4_mask_),
                   criterion(r_paf4, paf4_, paf4_mask_),
                   criterion(g_paf8, paf8_, paf8_mask_),
                   criterion(r_paf8, paf8_, paf8_mask_),
                   criterion(g_paf16, paf16_, paf16_mask_),
                   criterion(r_paf16, paf16_, paf16_mask_)]
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
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--prefix', type=str, default='default', help='model description')
    parser.add_argument('--version', type=int, default=2, choices=[2, 3], help='model version')
    parser.add_argument('--num-stage', type=int, default=3)
    parser.add_argument('--num-channel', type=int, default=256)
    parser.add_argument('--num-context', type=int, default=2)
    parser.add_argument('--kps-model', type=str, default='')
    args = parser.parse_args()
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
    num_context = args.num_context
    version = args.version
    if version == 2:
        base_name = 'V2.%s-%s-S%d-C%d-C%d-BS%d-%s' % (prefix, backbone, num_stage, num_channel, num_context, batch_size, optim)
    elif version == 3:
        base_name = 'V3.%s-%s-C%d-BS%d-%s' % (prefix, backbone, num_channel, batch_size, optim)
    else:
        raise RuntimeError('no such version %d'%version)
    filename = './tmp/%s.log' % base_name
    logger = get_logger(fn=filename)
    logger.info(args)
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
            net = PoseNet(num_kps=num_kps, num_limb=num_limb, num_stage=num_stage, num_channel=num_channel, num_context=num_context)
            creator, featnames, fixed = cfg.BACKBONE_v2[backbone]
        elif version == 3:
            net = CascadePoseNet(num_kps=num_kps, num_limb=num_limb, num_channel=num_channel)
            creator, featnames, fixed = cfg.BACKBONE_v3[backbone]
        else:
            raise RuntimeError('no such version %d'%version)
        net.initialize(mx.init.Normal(), ctx=ctx)
    else:
        logger.info('Load net from %s', model_path)
        net = load_model(model_path, version=version)
    net.init_backbone(creator, featnames, fixed, pretrained=True)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    criterion = SumL2Loss()
    criterion.hybridize()
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
        rds = [Recorder('G-h-04', freq), Recorder('R-h-04', freq), \
               Recorder('G-h-08', freq), Recorder('R-h-08', freq), \
               Recorder('G-h-16', freq), Recorder('R-h-16', freq), \
               Recorder('G-p-04', freq), Recorder('R-p-04', freq), \
               Recorder('G-p-08', freq), Recorder('R-p-08', freq), \
               Recorder('G-p-16', freq), Recorder('R-p-16', freq)]
    else:
        raise RuntimeError('no such version %d'%version)
    # meta info
    global_step = 0
    # forward and backward
    if version == 2:
        forward_backward = forward_backward_v2
    elif version == 3:
        forward_backward = forward_backward_v3
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
            losses = forward_backward(net, criterion, ctx, packet, is_train=True)
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
            losses = forward_backward(net, criterion, ctx, packet, is_train=False)
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
