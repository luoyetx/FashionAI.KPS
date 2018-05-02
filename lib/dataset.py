from __future__ import print_function, division

import os
import argparse
import cv2
import mxnet as mx
from mxnet import gluon as gl
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa

from lib.config import cfg
from lib.utils import process_cv_img, reverse_to_cv_img, crop_patch
from lib.utils import draw_heatmap, draw_kps, draw_paf, draw_box

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from heatmap import putGaussianMaps, putVecMaps


def random_aug_img(img):
    seq = iaa.Sequential([
        iaa.Sometimes(0.2, iaa.SomeOf(1, [
            iaa.GaussianBlur(sigma=(0, 1))]),
            iaa.MedianBlur(k=[1, 3]),
            iaa.AverageBlur(k=[1, 2]),
        ),
        iaa.Sometimes(0.2, iaa.Grayscale(alpha=[0.8, 1.], from_colorspace='BGR')),
        iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
        iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-20, 20))),
        iaa.Sometimes(0.2, iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
        ], random_order=True)
    img = seq.augment_images(img[np.newaxis])[0]
    return img


def transform(img, kps, is_train=True, rot=True):
    height, width = img.shape[:2]
    # flip
    if np.random.rand() < 0.5 and is_train:
        img = cv2.flip(img, 1)
        kps[:, 0] = width - kps[:, 0]
        for i, j in cfg.LANDMARK_SWAP:
            tmp = kps[i].copy()
            kps[i] = kps[j]
            kps[j] = tmp
    # rotate
    if is_train and rot:
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
    center = (width // 2, height // 2)
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
    # image augment
    if is_train:
        img = random_aug_img(img)
    # fill missing
    kps[kps[:, 2] == -1, :2] = -1
    return img, kps


def get_label_v2(height, width, category, kps):
    stride = 8
    sigma = 7
    grid_x = width // stride
    grid_y = height // stride
    # heatmap and mask
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_kps = len(kps)
    heatmap = np.zeros((num_kps + 1, grid_y, grid_x))
    heatmap_mask = np.zeros_like(heatmap)
    for i, (x, y, v) in enumerate(kps):
        if i in landmark_idx and v != -1:
            heatmap_mask[i] = 1
            putGaussianMaps(heatmap[i], heatmap_mask[i], x, y, v, stride, sigma)
            # ht = heatmap[i]
            # ht = (ht * 255).astype(np.uint8)
            # ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
            # cv2.imshow('ht', cv2.resize(ht, (0, 0), ht, 4, 4))
            # ht = heatmap_mask[i]
            # ht = (ht * 255).astype(np.uint8)
            # ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
            # cv2.imshow('mask', cv2.resize(ht, (0, 0), ht, 4, 4))
            # cv2.waitKey(0)
    heatmap[-1] = heatmap[::-1].max(axis=0)
    heatmap_mask[-1] = 1
    # paf
    limb = cfg.PAF_LANDMARK_PAIR
    num_limb = len(limb)
    paf = np.zeros((2 * num_limb, grid_y, grid_x))
    paf_mask = np.zeros_like(paf)
    for idx, (idx1, idx2) in enumerate(limb):
        x1, y1, v1 = kps[idx1]
        x2, y2, v2 = kps[idx2]
        if v1 != -1 and v2 != -1:
            putVecMaps(paf[2*idx], paf[2*idx + 1], x1, y1, x2, y2, stride, cfg.HEATMAP_THRES)
            paf_mask[2*idx] = 1
            paf_mask[2*idx + 1] = 1
    # result
    return heatmap.astype('float32'), paf.astype('float32'), heatmap_mask.astype('float32'), paf_mask.astype('float32')


def get_label_v3(height, width, category, kps):
    strides = [4, 8, 16]
    sigmas = [7, 7, 7]
    heatmaps, masks = [], []
    for stride, sigma in zip(strides, sigmas):
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
        mask[-1] = 1
        heatmaps.append(heatmap.astype('float32'))
        masks.append(mask.astype('float32'))
    # result
    return heatmaps, masks


def get_border(shape, kps, expand=0):
    h, w = shape
    keep = kps[:, 2] != -1
    xmin = kps[keep, 0].min()
    xmax = kps[keep, 0].max()
    ymin = kps[keep, 1].min()
    ymax = kps[keep, 1].max()
    bh, bw = ymax - ymin, xmax - xmin
    xmin -= expand * bw
    xmax += expand * bw
    ymin -= expand * bh
    ymax += expand * bh
    xmin = max(min(int(xmin), w), 0)
    xmax = max(min(int(xmax), w), 0)
    ymin = max(min(int(ymin), h), 0)
    ymax = max(min(int(ymax), h), 0)
    return (xmin, ymin, xmax, ymax)


def get_label_v4(height, width, category, kps):
    stride = 8
    sigma = 7
    # heatmap and mask
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_kps = len(kps)
    h, w = height // stride, width // stride
    heatmap = np.zeros((num_kps + 1, h, w))
    heatmap_mask = np.zeros_like(heatmap)
    for i, (x, y, v) in enumerate(kps):
        if i in landmark_idx and v != -1:
            heatmap_mask[i] = 1
            putGaussianMaps(heatmap[i], heatmap_mask[i], x, y, v, stride, sigma)
    heatmap[-1] = heatmap[::-1].max(axis=0)
    heatmap_mask[-1] = 1
    # obj and mask
    obj = np.zeros((5, h, w))
    obj_mask = np.zeros_like(obj)
    cate_idx = cfg.CATEGORY.index(category)
    xmin, ymin, xmax, ymax = get_border((height, width), kps, expand=0.1)
    xmin = xmin // stride
    ymin = ymin // stride
    xmax = xmax // stride + 1
    ymax = ymax // stride + 1
    obj[cate_idx, ymin: ymax, xmin: xmax] = 1
    obj_mask[cate_idx] = 1
    # result
    return heatmap.astype('float32'), heatmap_mask.astype('float32'), obj.astype('float32'), obj_mask.astype('float32')


class FashionAIKPSDataSet(gl.data.Dataset):

    def __init__(self, df, version=2, is_train=True):
        self.img_dir = cfg.DATA_DIR
        self.is_train = is_train
        self.version = version
        if version == 5:
            self.version = 2
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
        if self.version == 2:
            heatmap, paf, mask_heatmap, mask_paf = get_label_v2(height, width, category, kps)
            return img, heatmap, paf, mask_heatmap, mask_paf
        elif self.version == 3:
            ht, mask = get_label_v3(height, width, category, kps)
            return img, ht[0], mask[0], ht[1], mask[1], ht[2], mask[2]
        else:
            heatmap, heatmap_mask, obj, obj_mask = get_label_v4(height, width, category, kps)
            return img, heatmap, heatmap_mask, obj, obj_mask

    def __len__(self):
        return len(self.img_lst)


class FashionAIDetDataSet(gl.data.Dataset):

    def __init__(self, df, is_train=True):
        self.img_dir = cfg.DATA_DIR
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
        img, kps = transform(img, kps, self.is_train, rot=False)
        # preprocess
        height, width = img.shape[:2]
        img = process_cv_img(img)
        self.cur_kps = kps  # for debug and show
        # get label
        xmin, ymin, xmax, ymax = get_border((height, width), kps, expand=0)
        cate_idx = cfg.CATEGORY.index(category)
        label = np.array([xmin, ymin, xmax, ymax, cate_idx], dtype='float32')
        return img, label

    def __len__(self):
        return len(self.img_lst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    print(args)
    version = args.version
    np.random.seed(0)
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'train.csv'))
    dataset = FashionAIKPSDataSet(df, version=version, is_train=args.type == 'train')
    print('DataSet Size', len(dataset))
    for idx, pack in enumerate(dataset):
        if version == 2:
            data, heatmap, paf, mask_heatmap, mask_paf = pack
            heatmap = heatmap[::-1].max(axis=0)
        elif version == 3:
            data, ht4, mask4, ht8, mask8, ht16, mask16 = pack
        else:
            data, heatmap, heatmap_mask, obj, obj_mask = pack

        img = reverse_to_cv_img(data)
        kps = dataset.cur_kps
        category = dataset.category[idx]

        if version == 2:
            dr1 = draw_heatmap(img, heatmap)
            dr2 = draw_paf(img, paf)
            dr3 = draw_kps(img, kps)
            cv2.imshow("heatmap", dr1)
            cv2.imshow("paf", dr2)
            cv2.imshow("kps", dr3)
        elif version == 3:
            dr1 = draw_heatmap(img, ht4.max(axis=0), resize_im=True)
            dr2 = draw_heatmap(img, ht8.max(axis=0), True)
            dr3 = draw_heatmap(img, ht16.max(axis=0), True)
            dr4 = draw_kps(img, kps)
            cv2.imshow('h4', dr1)
            cv2.imshow('h8', dr2)
            cv2.imshow('h16', dr3)
            cv2.imshow('kps', dr4)
        else:
            cate_idx = cfg.CATEGORY.index(category)
            dr1 = draw_heatmap(img, heatmap.max(axis=0))
            dr2 = draw_heatmap(img, obj[cate_idx])
            cv2.imshow('heatmap', dr1)
            cv2.imshow('obj', dr2)

        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    main()
