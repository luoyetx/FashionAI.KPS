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
from lib.utils import process_cv_img, reverse_to_cv_img, crop_patch, crop_patch_refine
from lib.utils import draw_heatmap, draw_kps, draw_paf, draw_box

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from heatmap import putGaussianMaps, putPafMaps


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


def transform(img, kps, is_train=True, random_rot=True, random_scale=True):
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
    if is_train and random_rot:
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
    scale = np.random.rand() * (cfg.SCALE_MAX - cfg.SCALE_MIN) + cfg.SCALE_MIN if is_train and random_scale else cfg.CROP_SIZE
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


def get_label(height, width, category, kps, stride, sigma=7, th=1):
    grid_x = width // stride
    grid_y = height // stride
    # heatmap and mask
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_kps = len(kps)
    ht = np.zeros(shape=(num_kps, grid_y, grid_x))
    ht_mask = np.zeros(shape=(num_kps, 1, 1))
    for i, (x, y, v) in enumerate(kps):
        if i in landmark_idx and v == 1:
            ht_mask[i] = 1
            putGaussianMaps(ht[i], x, y, v, stride, sigma)
            # tx, ty = int(x / stride), int(y / stride)
            # if tx >=0 and tx < grid_x and ty >= 0 and ty < grid_y:
            #     ht[i, ty, tx] = 1
            #     ht[i] = cv2.GaussianBlur(ht[i], (gk, gk), 0)
            #     am = ht[i].max()
            #     ht[i] /= am
    # paf and mask
    limb = cfg.PAF_LANDMARK_PAIR
    num_limb = len(limb)
    paf = np.zeros(shape=(num_limb, grid_y, grid_x))
    paf_mask = np.zeros(shape=(num_limb, 1, 1))
    for idx, (idx1, idx2) in enumerate(limb):
        x1, y1, v1 = kps[idx1]
        x2, y2, v2 = kps[idx2]
        if v1 != -1 and v2 != -1:
            putPafMaps(paf[idx], x1, y1, x2, y2, stride, th)
            paf_mask[idx] = 1
    return ht.astype('float32'), ht_mask.astype('float32'), paf.astype('float32'), paf_mask.astype('float32')


def get_label_v2(height, width, category, kps):
    stride = 8
    sigma = 7
    th = 1
    ht8, ht8_mask, paf8, paf8_mask = get_label(height, width, category, kps, stride, sigma, th)
    return ht8, ht8_mask, paf8, paf8_mask


def get_label_v3(height, width, category, kps):
    strides = [16, 8, 4]
    sigmas = [7, 7, 7]
    ths = [1, 1, 1]
    rs = []
    for stride, sigma, th in zip(strides, sigmas, ths):
        r = get_label(height, width, category, kps, stride, sigma, th)
        rs.append(r)
    return rs


def get_label_patch(height, width, category, kps):
    stride = 1
    sigma = 7
    th = 1
    ht1, ht1_mask, paf1, paf1_mask = get_label(height, width, category, kps, stride, sigma, th)

    return ht1, ht1_mask


class FashionAIKPSDataSet(gl.data.Dataset):

    def __init__(self, df, version=2, is_train=True):
        self.img_dir = cfg.DATA_DIR
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
        if self.version == 2:
            ht8, ht8_mask, paf8, paf8_mask = get_label_v2(height, width, category, kps)
            return img, ht8, ht8_mask, paf8, paf8_mask
        else:
            r16, r8, r4 = get_label_v3(height, width, category, kps)
            ht4, ht4_mask, paf4, paf4_mask = r4
            ht8, ht8_mask, paf8, paf8_mask = r8
            ht16, ht16_mask, paf16, paf16_mask = r16
            return img, ht4, ht4_mask, paf4, paf4_mask, ht8, ht8_mask, paf8, paf8_mask, ht16, ht16_mask, paf16, paf16_mask

    def __len__(self):
        return len(self.img_lst)


class FashionAIPatchDataSet(gl.data.Dataset):

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
        img, kps = transform(img, kps, self.is_train)
        # preprocess
        self.cur_kps = kps  # for debug and show
        # get label
        height, width = img.shape[:2]
        ht1, _ = get_label_patch(height, width, category, kps)
        num_kps = len(kps)
        delta = 12
        offset = np.random.uniform(low=-delta, high=delta, size=(num_kps, 2))
        center = kps.copy()
        center[:, :2] -= offset
        data, ht, mask = crop_patch_refine(img, center, size=40, ht1=ht1)
        ht = ht.astype('float32')
        mask = mask.astype('float32')
        data = data.astype('float32')
        return data, ht, mask

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
        img, kps = transform(img, kps, self.is_train, random_rot=False, random_scale=True)
        # preprocess
        height, width = img.shape[:2]
        img = process_cv_img(img)
        self.cur_kps = kps  # for debug and show
        # get label
        xmin, ymin, xmax, ymax = get_border((height, width), kps, expand=0)
        cate_idx = cfg.DET_CATE[category]
        label = np.array([xmin, ymin, xmax, ymax, cate_idx], dtype='float32')
        return img, label

    def __len__(self):
        return len(self.img_lst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kps', choices=['kps', 'patch'])
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    print(args)
    np.random.seed(0)
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'train.csv'))
    if args.dataset == 'kps':
        dataset = FashionAIKPSDataSet(df, version=3, is_train=args.type == 'train')
        print('DataSet Size', len(dataset))
        for idx, packet in enumerate(dataset):
            # unpack
            data, ht4, ht4_mask, paf4, paf4_mask, ht8, ht8_mask, paf8, paf8_mask, ht16, ht16_mask, paf16, paf16_mask = packet

            img = reverse_to_cv_img(data)
            kps = dataset.cur_kps
            category = dataset.category[idx]
            cv2.imshow("kps", draw_kps(img, kps))
            cv2.imshow('h4', draw_heatmap(img, ht4.max(axis=0)))
            cv2.imshow('h8', draw_heatmap(img, ht8.max(axis=0)))
            cv2.imshow('h16', draw_heatmap(img, ht16.max(axis=0)))
            cv2.imshow('paf4', draw_paf(img, paf4))
            cv2.imshow('paf8', draw_paf(img, paf8))
            cv2.imshow('paf16', draw_paf(img, paf16))

            key = cv2.waitKey(0)
            if key == 27:
                break
    else:
        dataset = FashionAIPatchDataSet(df, is_train=args.type == 'train')
        print('DataSet Size', len(dataset))
        for idx, packet in enumerate(dataset):
            # unpack
            data, ht, mask = packet
            num_kps = cfg.NUM_LANDMARK
            for i in range(num_kps):
                if mask[i] != 0:
                    patch = reverse_to_cv_img(data[3*i:3*i+3]).copy()
                    cv2.imshow('%d'%i, draw_heatmap(patch, ht[i]))
            key = cv2.waitKey(0)
            if key == 27:
                break


if __name__ == '__main__':
    main()
