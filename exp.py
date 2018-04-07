from __future__ import print_function, division

import os
import argparse
import cv2
import mxnet as mx
from mxnet import gluon as gl, autograd as ag
from mxnet.gluon import nn
import pandas as pd
import numpy as np

from config import cfg
from utils import process_cv_img, reverse_to_cv_img, crop_patch, get_logger
from utils import draw_heatmap, draw_kps, draw_paf

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from heatmap import putGaussianMaps, putVecMaps


def get_center(kps):
    keep = kps[:, 2] != 0
    xmin = kps[keep, 0].min()
    xmax = kps[keep, 0].max()
    ymin = kps[keep, 1].min()
    ymax = kps[keep, 1].max()
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    return (xc, yc)


def transform(img, kps, is_train=True):
    height, width = img.shape[:2]
    # flip
    if np.random.rand() > 0.5 and is_train:
        img = cv2.flip(img, 1)
        kps[:, 0] = width - kps[:, 0]
        for i, j in cfg.LANDMARK_SWAP:
            tmp = kps[i].copy()
            kps[i] = kps[j]
            kps[j] = tmp
    # rotate
    if is_train:
        angle = (np.random.random() - 0.5) * 2 * cfg.ROT_MAX
        center = (width // 2, height // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1)
        cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
        dsize = (int(height * sin + width * cos), int(height * cos + width * sin))
        rot[0, 2] += dsize[0] // 2 - center[0]
        rot[1, 2] += dsize[1] // 2 - center[1]
        img = cv2.warpAffine(img, rot, dsize, img, borderMode=cv2.BORDER_CONSTANT, borderValue=cfg.FILL_VALUE)
        height, width = img.shape[:2]
        xy = kps[:, :2]
        xy = rot.dot(np.hstack([xy, np.ones((len(xy), 1))]).T).T
        kps[:, :2] = xy
    # scale
    scale = np.random.rand() * (cfg.SCALE_MAX - cfg.SCALE_MIN) + cfg.SCALE_MIN if is_train else cfg.CROP_SIZE
    max_edge = max(height, width)
    scale_factor = scale / max_edge
    img = cv2.resize(img, (0, 0), img, scale_factor, scale_factor)
    height, width = img.shape[:2]
    kps[:, :2] *= scale_factor
    # crop
    rand_x = (np.random.rand() - 0.5) * 2 * cfg.CROP_CENTER_OFFSET_MAX if is_train else 0
    rand_y = (np.random.rand() - 0.5) * 2 * cfg.CROP_CENTER_OFFSET_MAX if is_train else 0
    center = get_center(kps)
    center = (center[0] + rand_x, center[1] + rand_y)
    x1 = int(center[0] - cfg.CROP_SIZE / 2)
    y1 = int(center[1] - cfg.CROP_SIZE / 2)
    x2 = x1 + cfg.CROP_SIZE
    y2 = y1 + cfg.CROP_SIZE
    roi = (x1, y1, x2, y2)
    img = crop_patch(img, roi)
    height, width = img.shape[:2]
    kps[:, 0] -= x1
    kps[:, 1] -= y1
    # fill missing
    kps[kps[:, 2] == -1, :2] = -1
    return img, kps


def get_label_v3(height, width, category, kps):
    stride = 8
    sigma = 7
    # heatmap and mask
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_kps = len(kps)
    h, w = height // stride, width // stride
    heatmap = np.zeros((num_kps + 1, h, w))
    mask = np.zeros_like(heatmap)
    for i, (x, y, v) in enumerate(kps):
        if i in landmark_idx and v != -1:
            mask[i] = 1
            putGaussianMaps(heatmap[i], mask[i], x, y, v, stride, sigma)
    heatmap[-1] = heatmap[::-1].max(axis=0)
    # result
    return heatmap.astype('float32')


class FashionAIKPSDataSet(gl.data.Dataset):

    def __init__(self, df, version=2, is_train=True):
        self.img_dir = os.path.join(cfg.DATA_DIR, 'train')
        self.is_train = is_train
        self.version = version
        # img path
        self.img_lst = df['image_id'].tolist()
        self.category = df['image_category'].tolist()
        # kps, (x, y, v) v -> (not exists -1, occur 0, normal 1)
        cols = df.columns[2:]
        kps = []
        for i in range(cfg.NUM_LANDMARK):
            for j in range(3):
                kps.append(df[cols[i]].apply(lambda x: int(x.split('_')[j])).as_matrix())
        kps = np.vstack(kps).T.reshape((len(self.img_lst), -1, 3)).astype(np.float)
        self.kps = kps

    def __getitem__(self, idx):
        # meta
        img_path = os.path.join(self.img_dir, self.img_lst[idx])
        img = cv2.imread(img_path)
        category = self.category[idx]
        kps = self.kps[idx].copy()
        # transform
        img, kps = transform(img, kps, self.is_train)
        # preprocess
        height, width = img.shape[:2]
        img = process_cv_img(img)
        self.cur_kps = kps  # for debug and show
        # get label
        ht = get_label_v3(height, width, category, kps)
        mask = np.zeros((24, 2), dtype='float32')
        landmark_idx = cfg.LANDMARK_IDX[category]
        mask[landmark_idx] = 1
        mask = mask.reshape((24*2))
        kps = kps[:, :2]
        kps[:, 0] /= width
        kps[:, 1] /= height
        kps = kps.reshape((24*2)).astype('float32')
        return ht, kps, mask

    def __len__(self):
        return len(self.img_lst)


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

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = gl.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = gl.loss._apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.sum(loss, axis=self._batch_axis, exclude=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--freq', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print(args)
    bs = args.batch_size
    ctx = mx.cpu(0) if args.gpu < 0 else mx.gpu(args.gpu)
    freq = args.freq
    lr = args.lr

    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.MaxPool2D(5, 2))
        net.add(nn.Dense(256, activation='relu'))
        net.add(nn.Dense(24*2))
    net.initialize(mx.init.Normal(), ctx)
    net.hybridize()

    np.random.seed(0)
    df_train = pd.read_csv(os.path.join('./data/train.csv'))
    df_test = pd.read_csv(os.path.join('./data/val.csv'))
    traindata = FashionAIKPSDataSet(df_train, version=2, is_train=True)
    testdata = FashionAIKPSDataSet(df_test, version=2, is_train=False)
    trainloader = gl.data.DataLoader(traindata, batch_size=bs, shuffle=True, last_batch='discard')
    testloader = gl.data.DataLoader(testdata, batch_size=bs, shuffle=False, last_batch='discard')

    trainer = gl.trainer.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': 1e-5})
    criterion = SumL2Loss()
    logger = get_logger()
    rd = Recorder('L2', freq)

    for epoch_idx in range(100):
        # train part
        logger.info('[Epoch %d]', epoch_idx + 1)
        rd.reset()
        for batch_idx, (ht, kps, mask) in enumerate(trainloader):
            # [(l1, l2, ...), (l1, l2, ...)]
            ht = ht.as_in_context(ctx)
            kps = kps.as_in_context(ctx)
            mask = mask.as_in_context(ctx)
            with ag.record():
                kps_pred = net(ht)
                loss = criterion(kps_pred * mask, kps * mask)
                ag.backward(loss)
            trainer.step(bs)
            # reduce to [l1, l2, ...]
            rd.update(loss.mean().asscalar())
            if batch_idx % freq == freq - 1:
                name, value = rd.get()
                logger.info('[Epoch %d][Batch %d] %s = %f', epoch_idx + 1, batch_idx + 1, name, value)
        # test part
        rd.reset()
        for batch_idx, (ht, kps, mask) in enumerate(testloader):
            ht = ht.as_in_context(ctx)
            kps = kps.as_in_context(ctx)
            mask = mask.as_in_context(ctx)
            kps_pred = net(ht)
            loss = criterion(kps_pred * mask, kps * mask)
            rd.update(loss.mean().asscalar())
        name, value = rd.get()
        logger.info('[Epoch %d][Test] %s = %f', epoch_idx + 1, name, value)
        net.save_params('./output/kps_on_ht.params')


if __name__ == '__main__':
    main()
