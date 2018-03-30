from __future__ import print_function, division

import os
import cv2
import mxnet as mx
from mxnet import gluon as gl
import pandas as pd
import numpy as np
from config import cfg

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from heatmap import putGaussianMaps, putVecMaps


def crop_patch(img, bbox, wrap=True):
    height, width = img.shape[:-1]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
        print('[WARN] ridiculous x1, y1, x2, y2')
        return None
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        # out of boundary, still crop the face
        if not wrap:
            return None
        h, w = y2 - y1, x2 - x1
        patch = np.zeros((h, w, 3), dtype=np.uint8)
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


def transform(img, category, kps, is_train=True):
    height, width = img.shape[:2]
    miss = np.logical_or(kps[:, 0] == -1, kps[:, 1] == -1)
    update_miss = lambda kps: np.logical_or.reduce([kps[:, 0] < 0, kps[:, 1] < 0, kps[:, 0] > width, kps[:, 1] > height, miss])
    # flip
    if np.random.rand() > 0.5 and is_train:
        img = cv2.flip(img, 1)
        kps[:, 0] = width - kps[:, 0]
        for i, j in cfg.LANDMARK_SWAP[category]:
            tmp = kps[i].copy()
            kps[i] = kps[j]
            kps[j] = tmp
    # rotate
    if np.random.rand() > 0.5 and is_train:
        angle = (np.random.random() - 0.5) * 2 * cfg.ROT_MAX
        center = (width // 2, height // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1)
        cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
        dsize = (int(height * sin + width * cos), int(height * cos + width * sin))
        rot[0, 2] += dsize[0] // 2 - center[0]
        rot[1, 2] += dsize[1] // 2 - center[1]
        img = cv2.warpAffine(img, rot, dsize)
        height, width = img.shape[:2]
        kps = rot.dot(np.hstack([kps, np.ones((len(kps), 1))]).T).T
        miss = update_miss(kps)
    # scale
    scale = np.random.rand() * (cfg.SCALE_MAX - cfg.SCALE_MIN) + cfg.SCALE_MIN if is_train else cfg.CROP_SIZE
    max_edge = max(height, width)
    scale_factor = scale / max_edge
    img = cv2.resize(img, (0, 0), img, scale_factor, scale_factor)
    height, width = img.shape[:2]
    kps *= scale_factor
    # crop
    rand_x = (np.random.rand() - 0.5) * 2 * cfg.CROP_CENTER_OFFSET_MAX if is_train else 0
    rand_y = (np.random.rand() - 0.5) * 2 * cfg.CROP_CENTER_OFFSET_MAX if is_train else 0
    center = (width // 2, height // 2)
    x1 = int(center[0] - cfg.CROP_SIZE / 2)
    y1 = int(center[1] - cfg.CROP_SIZE / 2)
    x2 = x1 + cfg.CROP_SIZE
    y2 = y1 + cfg.CROP_SIZE
    roi = (x1, y1, x2, y2)
    img = crop_patch(img, roi)
    height, width = img.shape[:2]
    kps[:, 0] -= x1
    kps[:, 1] -= y1
    miss = update_miss(kps)
    # fill kissing
    kps[miss] = -1
    return img, kps


def get_label(img, category, kps):
    stride = cfg.STRIDE
    height, width = img.shape[:2]
    grid_x = width // stride
    grid_y = height // stride
    num_kps = len(kps)
    limb = cfg.PAF_LANDMARK_PAIR[category]
    num_limb = len(limb)
    heatmap = np.zeros((num_kps, grid_y, grid_x))
    paf = np.zeros((2 * num_limb, grid_y, grid_x))
    count = np.zeros((grid_y, grid_x))
    miss = np.logical_or(kps[:, 0] == -1, kps[:, 1] == -1)
    for i, (x, y) in enumerate(kps):
        if not miss[i]:
            putGaussianMaps(heatmap[i], height, width, x, y, stride, grid_x, grid_y, cfg.SIGMA)
    for i, (idx1, idx2) in enumerate(limb):
        if not miss[idx1] and not miss[idx2]:
            ldm1 = kps[idx1]
            ldm2 = kps[idx2]
            putVecMaps(paf[2 * i], paf[2 * i + 1], count, ldm1[0], ldm1[1], ldm2[0], ldm2[1], stride, grid_x, grid_y, cfg.SIGMA, cfg.THRE)
    return heatmap, paf


def process_cv_img(img):
    # HWC -> CHW, BGR -> RGB
    img = img.astype('float32').transpose((2, 0, 1))
    img = img[::-1, :, :]
    mean = np.array(cfg.PIXEL_MEAN, dtype='float32').reshape((3, 1, 1))
    std = np.array(cfg.PIXEL_STD, dtype='float32').reshape((3, 1, 1))
    img = (img - mean) / std
    return img


class FashionAIKPSDataSet(gl.data.Dataset):

    def __init__(self, df, category='blouse', is_train=True):
        df = df[df['image_category'] == category]
        df = df.sample(frac=1).reset_index(drop=True)
        self.img_dir = os.path.join(cfg.IMG_DIR, 'train')
        self.category = category
        self.category_idx = cfg.CATEGORY_2_IDX[category]
        self.is_train = is_train
        # img path
        self.img_lst = df['image_id'].tolist()
        # kps
        cols = df.columns[2:]
        kps = []
        for i in cfg.LANDMARK_IDX[category]:
            for j in range(2):
                kps.append(df[cols[i]].apply(lambda x: int(x.split('_')[j])).as_matrix())
        kps = np.vstack(kps).T.reshape((len(self.img_lst), -1, 2)).astype(np.float)
        self.kps = kps

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_lst[idx])
        img = cv2.imread(img_path)
        kps = self.kps[idx].copy()
        # t = draw_kps(img.copy(), kps)
        # cv2.imshow('ori', t)
        img, kps = transform(img, self.category, kps, self.is_train)
        heatmap, paf = get_label(img, self.category, kps)
        img = process_cv_img(img)
        heatmap = heatmap.astype('float32')
        paf = paf.astype('float32')
        return img, heatmap, paf

    def __len__(self):
        return len(self.img_lst)


def draw_kps(im, kps):
    im = im.copy()
    for (x, y) in kps:
        x, y = int(x), int(y)
        cv2.circle(im, (x, y), 2, (0, 0, 255), -1)
    return im


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('./data/train/Annotations/train.csv')
    dataset = FashionAIKPSDataSet(df, 'skirt')
    mean = np.array(cfg.PIXEL_MEAN, dtype='float32').reshape((3, 1, 1))
    std = np.array(cfg.PIXEL_STD, dtype='float32').reshape((3, 1, 1))
    np.random.seed(0)
    print(len(dataset))
    for (data, heatmap, paf) in dataset:
        data, heatmap, paf = dataset[10]
        img = (data * std + mean).astype('uint8').transpose((1, 2, 0))[:, :, ::-1]
        heatmap = heatmap.max(axis=0)
        paf = paf.max(axis=0)

        def draw(img, ht):
            ht = cv2.resize(ht, (0, 0), ht, 8, 8)
            ht = (ht * 255).astype(np.uint8)
            ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
            drawed = cv2.addWeighted(img, 0.5, ht, 0.5, 0)
            return drawed

        dr1 = draw(img, heatmap)
        dr2 = draw(img, paf)
        cv2.imshow("heatmap", dr1)
        cv2.imshow("paf", dr2)
        key = cv2.waitKey(0)
        if key == 27:
            break
