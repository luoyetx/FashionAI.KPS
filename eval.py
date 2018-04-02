from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import cv2
import mxnet as mx
import numpy as np
import pandas as pd

from dataset import FashionAIKPSDataSet
from model import PoseNet
from config import cfg
from utils import draw_heatmap, draw_paf, draw_kps
from utils import process_cv_img, detect_kps, get_logger, load_model, mkdir


def calc_error(kps_pred, kps_gt, category):
    dist = lambda dx, dy: np.sqrt(np.square(dx) + np.square(dy))
    idx1, idx2 = cfg.EVAL_NORMAL_IDX[category]
    norm = dist(kps_gt[idx1, 0] - kps_gt[idx2, 0], kps_gt[idx2, 1] - kps_gt[idx2, 1])
    if norm == 0:
        return -1
    keep = kps_gt[:, 2] == 1
    kps_gt = kps_gt[keep]
    kps_pred = kps_pred[keep]
    error = dist(kps_pred[:, 0] - kps_gt[:, 0], kps_pred[:, 1] - kps_gt[:, 1])
    error[kps_pred[:, 2] == -1] = norm  # fill missing with norm, so error = 1
    error /= norm
    error = error.mean()
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    print(args)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = cfg.DATA_DIR
    show = args.show
    save = args.save
    logger = get_logger()
    # model
    net = load_model(args.model)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # data
    df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    testdata = FashionAIKPSDataSet(df, False)
    num = len(testdata)
    base_name = './result/val'
    mkdir(base_name)
    for c in cfg.CATEGORY:
        mkdir('%s/%s' % (base_name, c))
    result = []
    for i in range(num):
        path = os.path.join(data_dir, 'train', testdata.img_lst[i])
        img = cv2.imread(path)
        category = testdata.category[i]
        kps_gt = testdata.kps[i]
        # predict
        data = process_cv_img(img)
        batch = mx.nd.array(data[np.newaxis], ctx)
        out = net(batch)
        heatmap = out[-1][0][0].asnumpy()
        paf = out[-1][1][0].asnumpy()
        # save output
        if save:
            out_path = '%s/%s/%s.npy' % (base_name, category, os.path.basename(path).split('.')[0])
            npy = np.concatenate([heatmap, paf])
            np.save(out_path, npy)
        # detect kps
        kps_pred = detect_kps(img, heatmap, paf, category)
        # calc_error
        error = calc_error(kps_pred, kps_gt, category)
        if error != -1:
            result.append(error)
        if i % 100 == 0:
            avg_error = np.array(result).mean()
            logger.info('Eval %d samples, Avg Normalized Error: %f', i + 1, avg_error)

        if show:
            landmark_idx = cfg.LANDMARK_IDX[category]
            htall = heatmap[-1]
            heatmap = heatmap[::-1].max(axis=0)

            dr1 = draw_heatmap(img, heatmap)
            dr2 = draw_paf(img, paf)
            dr3 = draw_kps(img, kps_pred)
            dr4 = draw_heatmap(img, htall)

            cv2.imshow('heatmap', dr1)
            cv2.imshow('paf', dr2)
            cv2.imshow('detect', dr3)
            cv2.imshow('htall', dr4)
            key = cv2.waitKey(0)
            if key == 27:
                break

    avg_error = np.array(result).mean()
    logger.info('Total Avg Normalized Error: %f', avg_error)
