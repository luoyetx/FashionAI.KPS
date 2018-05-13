from __future__ import print_function, division

import os
import logging
import datetime
import cv2
import mxnet as mx
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns

from lib.config import cfg


def process_cv_img(img):
    # HWC -> CHW, BGR -> RGB
    img = img.astype('float32').transpose((2, 0, 1)) / 255
    img = img[::-1, :, :]
    img = (img - cfg.PIXEL_MEAN) / cfg.PIXEL_STD
    return img

def reverse_to_cv_img(data):
    img = ((data * cfg.PIXEL_STD + cfg.PIXEL_MEAN) * 255).astype('uint8')
    img = img.transpose((1, 2, 0))[:, :, ::-1]
    return img


def crop_patch(img, bbox, fill_value=cfg.FILL_VALUE):
    height, width = img.shape[:-1]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
        # print('[WARN] ridiculous x1, y1, x2, y2')
        return None
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        # out of boundary, still crop the face
        h, w = y2 - y1, x2 - x1
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        patch[:, :] = fill_value
        vx1 = 0 if x1 < 0 else x1
        vy1 = 0 if y1 < 0 else y1
        vx2 = width if x2 > width else x2
        vy2 = height if y2 > height else y2
        sx = -x1 if x1 < 0 else 0
        sy = -y1 if y1 < 0 else 0
        vw = vx2 - vx1
        vh = vy2 - vy1
        patch[sy:sy+vh, sx:sx+vw] = img[vy1:vy2, vx1:vx2]
        return patch
    return img[y1:y2, x1:x2]


def crop_patch_refine(img, kps, size):
    num_kps = len(kps)
    data = np.zeros(shape=(3*num_kps, size, size))
    mask = np.zeros(shape=(num_kps, 2))
    l = size // 2
    for i, (x, y, v) in enumerate(kps):
        if v != -1:
            x, y = int(x), int(y)
            bbox = (x - l, y - l, x + l, y + l)
            patch = crop_patch(img, bbox)
            if patch is not None:
                patch = process_cv_img(patch)
                data[3*i:3*i+3] = patch
                mask[i] = 1
    return data, mask


def draw_kps(im, kps):
    im = im.copy()
    num_kps = len(kps)
    palette = np.array(sns.color_palette("hls", num_kps))
    palette = (palette * 255).astype('uint8')[:, ::-1].tolist()
    for idx, (x, y, v) in enumerate(kps):
        x, y = int(x), int(y)
        if v == 0:
            color = (0, 0, 0)
        elif v == 1:
            color = palette[idx]
        if v != -1:
            cv2.circle(im, (x, y), 3, color, -1)
            cv2.putText(im, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    return im


def draw_heatmap(im, ht, resize_im=False):
    assert len(ht.shape) == 2, ht.shape
    if resize_im:
        h, w = ht.shape
        im = cv2.resize(im, (w, h))
    else:
        h, w = im.shape[:2]
        ht = cv2.resize(ht, (w, h))
        ht[ht < 0] = 0
        ht[ht > 1] = 1
    ht = (ht * 255).astype(np.uint8)
    ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
    drawed = cv2.addWeighted(im, 0.5, ht, 0.5, 0)
    return drawed


def draw_paf(im, paf):
    assert len(paf.shape) == 3, paf.shape
    paf = paf.max(axis=0)
    return draw_heatmap(im, paf)


def draw_box(im, box, text=None):
    im = im.copy()
    x1, y1, x2, y2 = [int(_) for _ in box[:4]]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if text:
        cv2.putText(im, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    return im


def draw_det(im, det, draw_cate=None):
    im = im.copy()
    num_category = len(cfg.CATEGORY)
    palette = np.array(sns.color_palette("hls", num_category))
    palette = (palette * 255).astype('uint8')[:, ::-1].tolist()
    for i in range(num_category):
        if draw_cate and cfg.CATEGORY[i] != draw_cate:
            continue
        category = cfg.CATEGORY[i]
        color = palette[i]
        for proposal in det[i]:
            x1, y1, x2, y2 = [int(_) for _ in proposal[:4]]
            score = proposal[-1]
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
            cv2.putText(im, '%s_%0.2f' % (category, score), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    return im


def get_logger(name=None, fn=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    # file
    if fn:
        fh = logging.FileHandler(fn, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


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
