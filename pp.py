from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import cv2
import mxnet as mx
import numpy as np
import pandas as pd

from lib.model import load_model, multi_scale_predict
from lib.utils import draw_heatmap, draw_kps, draw_paf, crop_patch_refine
from lib.detect_kps import detect_kps
from lib.config import cfg


def calc_error(kps_pred, kps_gt, category):
    dist = lambda dx, dy: np.sqrt(np.square(dx) + np.square(dy))
    idx1, idx2 = cfg.EVAL_NORMAL_IDX[category]
    if kps_gt[idx1, 2] == -1 or kps_gt[idx2, 2] == -1:
        return 0, False
    norm = dist(kps_gt[idx1, 0] - kps_gt[idx2, 0], kps_gt[idx1, 1] - kps_gt[idx2, 1])
    keep = kps_gt[:, 2] == 1
    kps_gt = kps_gt[keep]
    kps_pred = kps_pred[keep]
    if keep.sum() == 0:
        # all occ
        return 0, False
    error = dist(kps_pred[:, 0] - kps_gt[:, 0], kps_pred[:, 1] - kps_gt[:, 1])
    error[kps_pred[:, 2] == -1] = norm  # fill missing with norm, so error = 1
    error = error / norm
    return error, True


def read_csv(path):
    df = pd.read_csv(path)
    # img path
    img_lst = df['image_id'].tolist()
    category = df['image_category'].tolist()
    # kps, (x, y, v) v -> (not exists -1, occur 0, normal 1)
    cols = df.columns[2:]
    kps = []
    for i in range(cfg.NUM_LANDMARK):
        for j in range(3):
            kps.append(df[cols[i]].apply(lambda x: int(x.split('_')[j])).as_matrix())
    kps = np.vstack(kps).T.reshape((len(img_lst), -1, 3)).astype(np.float)
    return img_lst, kps, category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='./data/val.csv')
    parser.add_argument('--pred', type=str, default='./result/val_result.csv')
    parser.add_argument('--th', type=float, default=0.05)
    parser.add_argument('--model', type=str)
    parser.add_argument('--version', type=int, default=2)
    args = parser.parse_args()
    print(args)
    img_lst, kps_gt, category = read_csv(args.gt)
    _, kps_pred, _ = read_csv(args.pred)
    assert len(kps_gt) == len(kps_pred)

    # model
    if args.model:
        ctx = mx.gpu(0)
        net = load_model(args.model, version=args.version)
        net.collect_params().reset_ctx(ctx)
        net.hybridize()

    th = args.th

    num_category = len(cfg.CATEGORY)
    result = [[] for i in range(num_category)]
    for img_id, gt, pred, cate in zip(img_lst, kps_gt, kps_pred, category):
        cate_idx = cfg.CATEGORY.index(cate)
        err, state = calc_error(pred, gt, cate)
        if state:
            result[cate_idx].append(err)
            if args.model and err.mean() > th:
                # predict
                img = cv2.imread('./data/' + img_id)
                heatmap, paf = multi_scale_predict(net, ctx, img, True)
                pred = detect_kps(img, heatmap, paf, cate)
                print('-------------------------')
                keep = np.where(gt[:, 2] == 1)[0]
                for i1, i2 in zip(keep, err):
                    print(i1, i2, gt[i1, :2], pred[i1, :2])
                print('mean', err.mean())
                print('-------------------------')
                # show
                landmark_idx = cfg.LANDMARK_IDX[cate]
                heatmap = heatmap[landmark_idx].max(axis=0)
                cv2.imshow('heatmap', draw_heatmap(img, heatmap))
                cv2.imshow('kps_pred', draw_kps(img, pred))
                cv2.imshow('kps_gt', draw_kps(img, gt))
                cv2.imshow('paf', draw_paf(img, paf))
                key = cv2.waitKey(0)
                if key == 27:
                    break

    result = [np.hstack(_) for _ in result]
    for i in range(num_category):
        category = cfg.CATEGORY[i]
        err = result[i].mean()
        print('Average Error for %s: %f' % (category, err))
    result = np.hstack(result)
    err = result.mean()
    print('Total Average Error %f' % err)


if __name__ == '__main__':
    main()
