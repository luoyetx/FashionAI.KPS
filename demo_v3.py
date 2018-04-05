from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import cv2
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import pandas as pd

from dataset_v3 import FashionAIKPSDataSet
from model import load_model
from utils import reverse_to_cv_img, draw_heatmap, draw_paf, draw_kps, detect_kps_v3
from config import cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--test-idx', type=int, default=0)
    parser.add_argument('--version', type=int, default=3)
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = cfg.DATA_DIR
    test_idx = args.test_idx
    # model
    net = load_model(args.model, 3)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # data
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    testdata = FashionAIKPSDataSet(df_test, False)
    category = df_test['image_category'][test_idx]
    landmark_idx = cfg.LANDMARK_IDX[category]

    data, ht = testdata[test_idx][:2]
    kps_gt = testdata.cur_kps
    img = reverse_to_cv_img(data)
    heatmap = net.predict(img, ctx)
    kps_pred = detect_kps_v3(img, heatmap, None, category)

    # render
    dr1 = draw_heatmap(img, ht.max(axis=0))
    dr2 = draw_heatmap(img, heatmap.max(axis=0))
    dr3 = draw_kps(img, kps_pred)
    dr4 = draw_kps(img, kps_gt)

    cv2.imshow('ori_heatmap', dr1)
    cv2.imshow('pred_heatmap', dr2)
    cv2.imshow('pred_kps', dr3)
    cv2.imshow('ori_kps', dr4)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
