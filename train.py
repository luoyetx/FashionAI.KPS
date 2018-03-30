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


class SumLoss(gl.loss.Loss):

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(SumLoss, self).__init__(weight, batch_axis, **kwargs)

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
    parser.add_argument('--cpm-stages', type=int, default=5)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg19'])
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    # data
    df = pd.read_csv(os.path.join(args.data_dir, 'train/Annotations/train.csv'))
    df = df.sample(frac=1)
    train_num = int(len(df) * 0.9)
    df_train = df[:train_num]
    df_test = df[train_num:]
    traindata = FashionAIKPSDataSet(df_train, args.category, True)
    testdata = FashionAIKPSDataSet(df_test, args.category, False)
    trainloader = gl.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, last_batch='discard', num_workers=4)
    testloader = gl.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=False, last_batch='discard', num_workers=4)
    im = cv2.imread(os.path.join(cfg.IMG_DIR, cfg.TEST_IMAGE[args.category]))
    im_data = process_cv_img(im)
    im_data = mx.nd.array(im_data[np.newaxis], ctx=ctx)
    # model
    num_kps = len(cfg.LANDMARK_IDX[args.category])
    num_limb = len(cfg.PAF_LANDMARK_PAIR[args.category])
    net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=args.cpm_stages)
    creator, featname, fixed = cfg.BACKBONE[args.backbone]
    net.init_backbone(creator, featname, fixed)
    net.initialize(mx.init.Normal(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    criterion = SumLoss()
    # trainer
    trainer = gl.trainer.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': args.wd, 'momentum': 0.9})
    # logger
    log_dir = './log/%s-%s' % (args.category, args.backbone)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    rds = []
    for i in range(args.cpm_stages):
        rd1 = Recorder('loss-heatmap_%d'%i, args.freq)
        rd2 = Recorder('loss-paf_____%d'%i, args.freq)
        rds.append(rd1)
        rds.append(rd2)
    global_step = 0
    logger = get_logger()
    # train model
    for epoch_idx in range(args.epoches):
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
                for stage_j in range(args.cpm_stages):
                    losses.append(criterion(out[stage_j][0], heatmap))
                    losses.append(criterion(out[stage_j][1], paf))
            ag.backward(losses)
            trainer.step(args.batch_size)
            for rd, loss in zip(rds, losses):
                rd.update(loss.mean().asscalar())
            if batch_idx % args.freq == args.freq - 1:
                for rd in rds:
                    name, value = rd.get()
                    writer.add_scalar('train/' + name, value, global_step)
                    logger.info('[epoch %d][batch %d] %s = %f' % (epoch_idx + 1, batch_idx + 1, name, value))
                global_step += 1
                toc = time.time()
                speed = (batch_idx + 1) * args.batch_size / (toc - tic)
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
            for stage_j in range(args.cpm_stages):
                losses.append(criterion(out[stage_j][0], heatmap))
                losses.append(criterion(out[stage_j][1], paf))
            for rd, loss in zip(rds, losses):
                rd.update(loss.mean().asscalar())
        for rd in rds:
            name, value = rd.get()
            writer.add_scalar('test/' + name, value, global_step)
            logger.info('[epoch %d][test] %s = %f' % (epoch_idx + 1, name, value))
        # # render part
        # ht = net(im_data)
        # ht = ht[-1][0].asnumpy()
        # ht = ht.max(axis=0)
        # ht = cv2.resize(ht, (0, 0), ht, 8, 8)
        # ht = (ht * 255).astype(np.uint8)
        # ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
        # drawed = cv2.addWeighted(im, 0.5, ht, 0.5, 0)
        # save_path = './tmp/%s-%s-%04d.jpg' % (args.category, args.backbone, epoch_idx + 1)
        # cv2.imwrite(save_path, drawed)
        # save part
        save_path = './output/%s-%s-%04d.params' % (args.category, args.backbone, epoch_idx + 1)
        net.save_params(save_path)
