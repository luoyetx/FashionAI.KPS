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
from lib.model import PoseNet, load_model
from lib.utils import get_logger, Recorder


class SumL2Loss(gl.loss.Loss):

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(SumL2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gl.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = gl.loss._apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.sum(loss, axis=self._batch_axis, exclude=True)


def forward_backward(net, criterion, ctx, data, heatmap, paf, heatmap_mask, paf_mask, is_train=True):
    data = gl.utils.split_and_load(data, ctx)
    heatmap = gl.utils.split_and_load(heatmap, ctx)
    heatmap_mask = gl.utils.split_and_load(heatmap_mask, ctx)
    paf = gl.utils.split_and_load(paf ,ctx)
    paf_mask = gl.utils.split_and_load(paf_mask ,ctx)
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
            losses_.append(criterion(out_[i][0], heatmap_))
            losses_.append(criterion(out_[i][1], paf_))
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
    parser.add_argument('--iter-size', type=int, default=1)
    parser.add_argument('--freq', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--cpm-stages', type=int, default=5)
    parser.add_argument('--cpm-channels', type=int, default=64)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--steps', type=str, default='1000')
    parser.add_argument('--lr-decay', type=float, default=0.1)
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg16', 'vgg19', 'resnet50'])
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--prefix', type=str, default='default', help='model description')
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # hyper parameters
    ctx = [mx.gpu(int(x)) for x in args.gpu.split(',')]
    data_dir = cfg.DATA_DIR
    cpm_stages = args.cpm_stages
    cpm_channels = args.cpm_channels
    lr = args.lr
    wd = args.wd
    optim = args.optim
    batch_size = args.batch_size
    iter_size = args.iter_size
    assert iter_size == 1
    epoches = args.epoches
    freq = args.freq
    steps = [int(x) for x in args.steps.split(',')]
    lr_decay = args.lr_decay
    backbone = args.backbone
    start_epoch = args.start_epoch
    prefix = args.prefix
    model_path = None if args.model_path == '' else args.model_path
    base_name = 'V2.%s-%s-S%d-C%d-BS%d-%s' % (prefix, backbone, cpm_stages, cpm_channels, batch_size, optim)
    logger = get_logger()
    # data
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    traindata = FashionAIKPSDataSet(df_train, version=2, is_train=True)
    testdata = FashionAIKPSDataSet(df_test, version=2, is_train=False)
    trainloader = gl.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=4)
    testloader = gl.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=4)
    epoch_size = len(trainloader)
    # model
    if start_epoch == 1 and not model_path:
        num_kps = cfg.NUM_LANDMARK
        num_limb = len(cfg.PAF_LANDMARK_PAIR)
        net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
        creator, featname, fixed = cfg.BACKBONE_v2[backbone]
        net.init_backbone(creator, featname, fixed, pretrained=True)
        net.initialize(mx.init.Normal(), ctx=ctx)
    else:
        model_path = model_path or './output/%s-%04d.params' % (base_name, start_epoch - 1)
        logger.info('Load net from %s', model_path)
        net = load_model(model_path)
    net.collect_params().reset_ctx(ctx)
    # for p in net.collect_params().values():
    #     p.grad_req = 'add'
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
    log_dir = './log-v2/%s'%base_name
    if os.path.exists(log_dir) and start_epoch == 1:
        shutil.rmtree(log_dir)
    sw = SummaryWriter(log_dir)
    rds = []
    for i in range(cpm_stages):
        rd1 = Recorder('h-%d' % i, freq)
        rd2 = Recorder('p-%d' % i, freq)
        rds.append(rd1)
        rds.append(rd2)
    # meta info
    if start_epoch == 1:
        global_step = 0
    else:
        meta_path = './output/%s-%04d.meta' % (base_name, start_epoch - 1)
        logger.info('Load meta from %s', meta_path)
        global_step = pickle.load(open(meta_path, 'rb'))
    # update ctx
    ctx = ctx * iter_size
    for epoch_idx in range(start_epoch - 1, epoches):
        # train part
        tic = time.time()
        for rd in rds:
            rd.reset()
        sw.add_scalar('lr', trainer.learning_rate, global_step)
        for batch_idx, (data, heatmap, paf, heatmap_mask, paf_mask) in enumerate(trainloader):
            # [(l1, l2, ...), (l1, l2, ...)]
            net.collect_params().zero_grad()
            losses = forward_backward(net, criterion, ctx, data, heatmap, paf, heatmap_mask, paf_mask, is_train=True)
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
        for batch_idx, (data, heatmap, paf, heatmap_mask, paf_mask) in enumerate(testloader):
            losses = forward_backward(net, criterion, ctx, data, heatmap, paf, heatmap_mask, paf_mask, is_train=False)
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
        save_path = './output/%s-%04d.meta' % (base_name, epoch_idx + 1)
        pickle.dump(global_step, open(save_path, 'wb'))
        logger.info('[Epoch %d] Saved to %s', epoch_idx + 1, save_path)
        save_path = './output/%s-%04d.params' % (base_name, epoch_idx + 1)
        net.save_params(save_path)
        logger.info('[Epoch %d] Saved to %s', epoch_idx + 1, save_path)


if __name__ == '__main__':
    main()
