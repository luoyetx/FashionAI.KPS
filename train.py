from __future__ import print_function, division

import os
import time
import shutil
import logging
import argparse
import cv2
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from dataset import FashionAIKPSDataSet, process_cv_img
from model import PoseNet
from config import cfg


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


def get_logger(name=None):
    """return a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class SumL2Loss(gl.loss.Loss):

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(SumL2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gl.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = gl.loss._apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.sum(loss, axis=self._batch_axis, exclude=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='-1')
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--category', type=str, default='skirt', choices=['blouse', 'skirt', 'outwear', 'dress', 'trousers'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--cpm-stages', type=int, default=5)
    parser.add_argument('--cpm-channels', type=int, default=64)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--steps', type=str, default='30,60')
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg19'])
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    cpm_stages = args.cpm_stages
    cpm_channels = args.cpm_channels
    lr = args.lr
    wd = args.wd
    optim = args.optim
    batch_size = args.batch_size
    epoches = args.epoches
    freq = args.freq
    steps = [int(x) for x in args.steps.split(',')]
    category = args.category
    data_dir = args.data_dir
    backbone = args.backbone
    base_name = '%s-%s-S%d-C%d-%s' % (category, backbone, cpm_stages, cpm_channels, optim)
    # data
    df = pd.read_csv(os.path.join(data_dir, 'train/Annotations/train.csv'))
    df = df.sample(frac=1)
    train_num = int(len(df) * 0.9)
    df_train = df[:train_num]
    df_test = df[train_num:]
    traindata = FashionAIKPSDataSet(df_train, category, True)
    testdata = FashionAIKPSDataSet(df_test, category, False)
    trainloader = gl.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=4)
    testloader = gl.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=4)
    epoch_size = len(trainloader)
    # model
    num_kps = len(cfg.LANDMARK_IDX[category])
    num_limb = len(cfg.PAF_LANDMARK_PAIR[category])
    net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
    creator, featname, fixed = cfg.BACKBONE[backbone]
    net.init_backbone(creator, featname, fixed)
    net.initialize(mx.init.Normal(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    criterion = SumL2Loss()
    # trainer
    steps = [epoch_size * x for x in steps]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.1)
    if optim == 'sgd':
        trainer = gl.trainer.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd, 'momentum': 0.9, 'lr_scheduler': lr_scheduler})
    else:
        trainer = gl.trainer.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'lr_scheduler': lr_scheduler})
    # logger
    log_dir = './log/%s' % base_name
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    rds = []
    for i in range(cpm_stages):
        rd1 = Recorder('h-%d' % i, freq)
        rd2 = Recorder('p-%d' % i, freq)
        rds.append(rd1)
        rds.append(rd2)
    logger = get_logger()
    # train model
    global_step = 0
    for epoch_idx in range(epoches):
        # train part
        for rd in rds:
            rd.reset()
        tic = time.time()
        for batch_idx, (data, heatmap, paf) in enumerate(trainloader):
            data = data.as_in_context(ctx)
            heatmap = heatmap.as_in_context(ctx)
            paf = paf.as_in_context(ctx)
            with ag.record():
                out = net(data)
                losses = []
                for stage_j in range(cpm_stages):
                    losses.append(criterion(out[stage_j][0], heatmap))
                    losses.append(criterion(out[stage_j][1], paf))
            ag.backward(losses)
            trainer.step(batch_size)
            for rd, loss in zip(rds, losses):
                rd.update(loss.mean().asscalar())
            if batch_idx % freq == freq - 1:
                writer.add_scalar('lr', trainer.learning_rate, global_step)
                for rd in rds:
                    name, value = rd.get()
                    writer.add_scalar('train/' + name, value, global_step)
                    logger.info('[epoch %d][batch %d] %s = %f' % (epoch_idx + 1, batch_idx + 1, name, value))
                global_step += 1
                toc = time.time()
                speed = (batch_idx + 1) * batch_size / (toc - tic)
                logger.info('[epoch %d][batch %d] Speed = %.2f sample/sec' % (epoch_idx + 1, batch_idx + 1, speed))
        toc = time.time()
        logger.info('[epoch %d] Train Cost %.0f sec' % (epoch_idx + 1, toc - tic))
        # test part
        for rd in rds:
            rd.reset()
        for batch_idx, (data, heatmap, paf) in enumerate(testloader):
            data = data.as_in_context(ctx)
            heatmap = heatmap.as_in_context(ctx)
            paf = paf.as_in_context(ctx)
            out = net(data)
            losses = []
            for stage_j in range(cpm_stages):
                losses.append(criterion(out[stage_j][0], heatmap))
                losses.append(criterion(out[stage_j][1], paf))
            for rd, loss in zip(rds, losses):
                rd.update(loss.mean().asscalar())
        for rd in rds:
            name, value = rd.get()
            writer.add_scalar('test/' + name, value, global_step)
            logger.info('[epoch %d][test] %s = %f' % (epoch_idx + 1, name, value))
        # save part
        save_path = './output/%s-%04d.params' % (base_name, epoch_idx + 1)
        net.save_params(save_path)
