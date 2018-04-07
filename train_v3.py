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

from dataset import FashionAIKPSDataSet
from model import PoseNet, CascadePoseNet
from config import cfg
from utils import get_logger


class Recorder(object):

    def __init__(self, name, length=100):
        self._name = name
        self._length = length
        self._num = np.zeros(length)
        self._full = False
        self._index = 0
        self._sum = 0
        self._count = 0

    def reset(self):
        self._num = np.zeros(self._length)
        self._full = False
        self._index = 0
        self._sum = 0
        self._count = 0

    def update(self, x):
        self._sum += x
        self._count += 1
        self._num[self._index] = x
        self._index += 1
        if self._index >= self._length:
            self._full = True
            self._index = 0

    def get(self, recent=True):
        if recent:
            if self._full:
                val = self._num.mean()
            else:
                val = self._num[:self._index].mean()
        else:
            val = self._sum / self._count
        return (self._name, val)


class SumL2Loss(gl.loss.Loss):

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(SumL2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, mask, sample_weight=None):
        pred = F.elemwise_mul(pred, mask)
        label = F.elemwise_mul(label, mask)
        label = gl.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = gl.loss._apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.sum(loss, axis=self._batch_axis, exclude=True)


def forward_backward(net, criterion, ctx, data, ht, mask, is_train=True):
    n = len(ht)
    m = len(ctx)
    data = gl.utils.split_and_load(data, ctx)
    ht = [gl.utils.split_and_load(x, ctx) for x in ht]
    mask = [gl.utils.split_and_load(x, ctx) for x in mask]
    ag.set_recording(is_train)
    ag.set_training(is_train)
    losses = []
    for data_, ht4_, mask4_, ht8_, mask8_, ht16_, mask16_ in zip(data, ht[0], mask[0], ht[1], mask[1], ht[2], mask[2]):
        # forward
        global_pred, refine_pred = net(data_)
        # global
        losses_ = [criterion(global_pred[0], ht4_, mask4_),
                   criterion(global_pred[1], ht8_, mask8_),
                   criterion(global_pred[2], ht16_, mask16_),
                   criterion(refine_pred, ht4_, mask4_)]
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
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--num-channel', type=int, default=128)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--steps', type=str, default='1000')
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg16', 'vgg19', 'resnet50'])
    parser.add_argument('--prefix', type=str, default='default', help='model description')
    parser.add_argument('--fixbn', action='store_true')
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # hyper parameters
    ctx = [mx.gpu(int(x)) for x in args.gpu.split(',')]
    data_dir = cfg.DATA_DIR
    num_channel = args.num_channel
    lr = args.lr
    wd = args.wd
    optim = args.optim
    batch_size = args.batch_size
    iter_size = args.iter_size
    epoches = args.epoches
    freq = args.freq
    steps = [int(x) for x in args.steps.split(',')]
    backbone = args.backbone
    fixbn = args.fixbn
    prefix = args.prefix
    base_name = 'V3.%s-%s-C%d-BS%d-%s' % (prefix, backbone, num_channel, batch_size, optim)
    logger = get_logger()
    # data
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    traindata = FashionAIKPSDataSet(df_train, version=3, is_train=True)
    testdata = FashionAIKPSDataSet(df_test, version=3, is_train=False)
    trainloader = gl.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=4)
    testloader = gl.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=4)
    epoch_size = len(trainloader)
    # model
    num_kps = cfg.NUM_LANDMARK
    net = CascadePoseNet(num_kps=num_kps, num_channel=num_channel)
    creator, featname, fixed = cfg.BACKBONE_v3[backbone]
    net.init_backbone(creator, featname, fixed, pretrained=True, fixbn=fixbn)
    net.initialize(mx.init.Normal(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    criterion = SumL2Loss()
    net.hybridize()
    criterion.hybridize()
    # trainer
    steps = [epoch_size * x for x in steps]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.1)
    if optim == 'sgd':
        trainer = gl.trainer.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd, 'momentum': 0.9, 'lr_scheduler': lr_scheduler})
    else:
        trainer = gl.trainer.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'lr_scheduler': lr_scheduler})
    # logger
    log_dir = './log-v3/%s'%base_name
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    sw = SummaryWriter(log_dir)
    # net(mx.nd.zeros(shape=(1, 3, 368, 368), ctx=ctx[0]))
    # sw.add_graph(net)
    rds = [Recorder('Global-04', freq), Recorder('Global-08', freq), Recorder('Global-16', freq), Recorder('Refine', freq)]
    # meta info
    global_step = 0
    # update ctx
    ctx = ctx * iter_size
    for epoch_idx in range(epoches):
        # train part
        tic = time.time()
        for rd in rds:
            rd.reset()
        sw.add_scalar('lr', trainer.learning_rate, global_step)
        for batch_idx, (data, ht4, mask4, ht8, mask8, ht16, mask16) in enumerate(trainloader):
            # [(l1, l2, ...), (l1, l2, ...)]
            ht = [ht4, ht8, ht16]
            mask = [mask4, mask8, mask16]
            losses = forward_backward(net, criterion, ctx, data, ht, mask, is_train=True)
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
        for batch_idx, (data, ht4, mask4, ht8, mask8, ht16, mask16) in enumerate(testloader):
            ht = [ht4, ht8, ht16]
            mask = [mask4, mask8, mask16]
            losses = forward_backward(net, criterion, ctx, data, ht, mask, is_train=False)
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
