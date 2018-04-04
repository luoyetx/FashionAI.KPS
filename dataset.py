from __future__ import print_function, division

import os
import cv2
import mxnet as mx
from mxnet import gluon as gl
import pandas as pd
import numpy as np

from config import cfg
from utils import process_cv_img, reverse_to_cv_img, crop_patch, draw_heatmap, draw_kps, draw_paf

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


def get_label(img, category, kps):
    stride = cfg.STRIDE
    height, width = img.shape[:2]
    grid_x = width // stride
    grid_y = height // stride
    # heatmap and mask
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_kps = len(kps)
    heatmap = np.zeros((num_kps + 1, grid_y, grid_x))
    heatmap_mask = np.zeros_like(heatmap)
    for i, (x, y, v) in enumerate(kps):
        if i in landmark_idx:
            heatmap_mask[i] = 1
        if v != -1:
            putGaussianMaps(heatmap[i], height, width, x, y, stride, grid_x, grid_y, cfg.HEATMAP_SIGMA)
    heatmap[-1] = heatmap[::-1].max(axis=0)
    heatmap_mask[-1] = heatmap_mask[::-1].max(axis=0)
    # paf
    limb = cfg.PAF_LANDMARK_PAIR
    num_limb = len(limb)
    paf = np.zeros((2 * num_limb, grid_y, grid_x))
    paf_mask = np.zeros_like(paf)
    for idx, (idx1, idx2) in enumerate(limb):
        x1, y1, v1 = kps[idx1]
        x2, y2, v2 = kps[idx2]
        if v1 != -1 and v2 != -1:
            putVecMaps(paf[2*idx], paf[2*idx + 1], x1, y1, x2, y2, stride, grid_x, grid_y, cfg.HEATMAP_THRES)
            paf_mask[2*idx] = 1
            paf_mask[2*idx + 1] = 1
    # result
    return heatmap, paf, heatmap_mask, paf_mask


class FashionAIKPSDataSet(gl.data.Dataset):

    def __init__(self, df, is_train=True):
        self.img_dir = os.path.join(cfg.DATA_DIR, 'train')
        self.is_train = is_train
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
        heatmap, paf, mask_heatmap, mask_paf = get_label(img, category, kps)
        # preprocess
        img = process_cv_img(img)
        heatmap = heatmap.astype('float32')
        paf = paf.astype('float32')
        mask_heatmap = mask_heatmap.astype('float32')
        mask_paf = mask_paf.astype('float32')
        self.cur_kps = kps  # for debug and show
        return img, heatmap, paf, mask_heatmap, mask_paf

    def __len__(self):
        return len(self.img_lst)


def main():
    np.random.seed(0)
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'train.csv'))
    dataset = FashionAIKPSDataSet(df)
    print(len(dataset))
    for idx, (data, heatmap, paf, mask_heatmap, mask_paf) in enumerate(dataset):
        #data, heatmap, paf, mask_heatmap, mask_paf = dataset[10]
        img = reverse_to_cv_img(data)
        heatmap = heatmap[-1]
        kps = dataset.cur_kps
        category = dataset.category[idx]

        dr1 = draw_heatmap(img, heatmap)
        dr2 = draw_paf(img, paf)
        dr3 = draw_kps(img, kps)

        cv2.imshow("heatmap", dr1)
        cv2.imshow("paf", dr2)
        cv2.imshow("%s-kps"%category, dr3)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    main()
