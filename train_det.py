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
from lib.dataset import FashionAIDetDataSet
from lib.model import DetNet
from lib.rpn import AnchorProposal
from lib.utils import get_logger, Recorder


class RpnClsLoss(gl.loss.Loss):

    def __init__(self, axis=-1, num_anchors=128, weight=None, batch_axis=0, **kwargs):
        super(RpnClsLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._num_anchors = num_anchors

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        pred = F.reshape(F.transpose(pred, (0, 2, 3, 1)), (-1, 2))
        label = F.reshape(F.transpose(label, (0, 2, 3, 1)), (-1, 1))
        sample_weight = F.reshape(F.transpose(sample_weight, (0, 2, 3, 1)), (-1, 1))

        pred = F.log_softmax(pred, self._axis)
        loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        loss = gl.loss._apply_weighting(F, loss, self._weight, sample_weight)
        return F.sum(loss) / self._num_anchors

class RpnRegLoss(gl.loss.Loss):

    def __init__(self, batch_axis=0, **kwargs):
        super(RpnRegLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=3)
        return F.sum(loss)


def forward_backward(net, criterions, ctx, data, rois, is_train=True):
    criterion_cls1, criterion_cls2, criterion_reg = criterions
    data = gl.utils.split_and_load(data, ctx)
    rois = gl.utils.split_and_load(rois, ctx)
    ag.set_recording(is_train)
    ag.set_training(is_train)
    # forward rpn
    rpn_cls1, rpn_reg1, rpn_cls2, rpn_reg2 = [], [], [], []
    for data_ in data:
        rpn_cls1_, rpn_reg1_, rpn_cls2_, rpn_reg2_ = net(data_)
        rpn_cls1.append(rpn_cls1_)
        rpn_reg1.append(rpn_reg1_)
        rpn_cls2.append(rpn_cls2_)
        rpn_reg2.append(rpn_reg2_)
    losses = []
    anchor_proposals = net.anchor_proposals
    for data_, rois_, rpn_cls1_, rpn_reg1_, rpn_cls2_, rpn_reg2_ in zip(data, rois, rpn_cls1, rpn_reg1, rpn_cls2, rpn_reg2):
        im_info = data_.shape[-2:]
        # anchor target
        # feat 1/8
        # parallel stops here
        batch_label1, batch_label_weight1, batch_bbox_targets1, batch_bbox_weights1 = anchor_proposals[0].target(rpn_cls1_, rois_, im_info)
        # loss cls
        loss_cls1 = criterion_cls1(rpn_cls1_, batch_label1, batch_label_weight1) / data_.shape[0]
        # loss reg
        loss_reg1 = criterion_reg(rpn_reg1_, batch_bbox_targets1, batch_bbox_weights1) / data_.shape[0]
        # feat 1/16
        # parallel stops here
        batch_label2, batch_label_weight2, batch_bbox_targets2, batch_bbox_weights2 = anchor_proposals[1].target(rpn_cls2_, rois_, im_info)
        # loss cls
        loss_cls2 = criterion_cls2(rpn_cls2_, batch_label2, batch_label_weight2) / data_.shape[0]
        # loss reg
        loss_reg2 = criterion_reg(rpn_reg2_, batch_bbox_targets2, batch_bbox_weights2) / data_.shape[0]

        loss = [loss_cls1, loss_reg1, loss_cls2, loss_reg2]
        # backward
        if is_train:
            ag.backward(loss)
        losses.append(loss)
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--steps', type=str, default='1000')
    parser.add_argument('--lr-decay', type=float, default=0.1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'vgg19', 'resnet50'])
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
    prefix = args.prefix
    model_path = None if args.model_path == '' else args.model_path
    base_name = 'Det.%s-%s-BS%d-%s' % (prefix, backbone, batch_size, optim)
    logger = get_logger()
    # data
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    traindata = FashionAIDetDataSet(df_train, is_train=True)
    testdata = FashionAIDetDataSet(df_test, is_train=False)
    trainloader = gl.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=4)
    testloader = gl.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, last_batch='discard', num_workers=4)
    epoch_size = len(trainloader)
    # model
    feat_stride = cfg.FEAT_STRIDE
    scales = cfg.DET_SCALES
    ratios = cfg.DET_RATIOS
    anchor_per_sample = [256, 128]
    anchor_proposals = [AnchorProposal(scales[i], ratios, feat_stride[i], anchor_per_sample[i]) for i in range(2)]
    net = DetNet(anchor_proposals)
    creator, featname, fixed = cfg.BACKBONE_Det[backbone]
    net.init_backbone(creator, featname, fixed, pretrained=True)
    if model_path:
        net.load_params(model_path)
    else:
        net.initialize(mx.init.Normal(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    criterion_cls1 = RpnClsLoss(num_anchors=256)
    criterion_cls2 = RpnClsLoss(num_anchors=128)
    criterion_reg = RpnRegLoss()
    criterion_cls1.hybridize()
    criterion_cls2.hybridize()
    criterion_reg.hybridize()
    criterions = [criterion_cls1, criterion_cls2, criterion_reg]
    # trainer
    steps = [epoch_size * x for x in steps]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_decay)
    if optim == 'sgd':
        trainer = gl.trainer.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd, 'momentum': 0.9, 'lr_scheduler': lr_scheduler})
    else:
        trainer = gl.trainer.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'lr_scheduler': lr_scheduler})
    # logger
    log_dir = './log-det/%s'%base_name
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    sw = SummaryWriter(log_dir)
    rds = [Recorder('rpn-cls-08', freq), Recorder('rpn-reg-08', freq), Recorder('rpn-cls-16', freq), Recorder('rpn-reg-16', freq)]
    # meta info
    global_step = 0
    # update ctx
    for epoch_idx in range(0, epoches):
        # train part
        tic = time.time()
        for rd in rds:
            rd.reset()
        sw.add_scalar('lr', trainer.learning_rate, global_step)
        for batch_idx, (data, rois) in enumerate(trainloader):
            # [(l1, l2, ...), (l1, l2, ...)]
            net.collect_params().zero_grad()
            losses = forward_backward(net, criterions, ctx, data, rois, is_train=True)
            trainer.step(1)
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
        for batch_idx, (data, rois) in enumerate(testloader):
            losses = forward_backward(net, criterions, ctx, data, rois, is_train=False)
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
