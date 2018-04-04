from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import cv2
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import pandas as pd

from dataset import FashionAIKPSDataSet
from model import PoseNet, load_model
from utils import reverse_to_cv_img, draw_heatmap, draw_paf, draw_kps, parse_from_name, detect_kps
from config import cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--test-idx', type=int, default=0)
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
    net = load_model(args.model)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # data
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    testdata = FashionAIKPSDataSet(df_test, False)
    category = df_test['image_category'][test_idx]
    landmark_idx = cfg.LANDMARK_IDX[category]

    data, heatmap, paf, heatmap_mask, paf_mask = testdata[test_idx]
    kps_gt = testdata.cur_kps
    img = reverse_to_cv_img(data)
    out_heatmap, out_paf = net.predict(img, ctx)
    kps_pred = detect_kps(img, out_heatmap, out_paf, category)

    # render
    heatmap = heatmap[-1]
    out_htall = out_heatmap[-1]
    out_heatmap = out_heatmap[landmark_idx].max(axis=0)

    dr1 = draw_heatmap(img, heatmap)
    dr2 = draw_paf(img, paf)
    dr3 = draw_heatmap(img, out_heatmap)
    dr4 = draw_paf(img, out_paf)
    dr5 = draw_heatmap(img, out_htall)
    dr6 = draw_kps(img, kps_pred)
    dr7 = draw_kps(img, kps_gt)

    cv2.imwrite('./tmp/%s_%d_ori_heatmap.jpg' % (category, test_idx), dr1)
    cv2.imwrite('./tmp/%s_%d_ori_paf.jpg' % (category, test_idx), dr2)
    cv2.imwrite('./tmp/%s_%d_pred_heatmap.jpg' % (category, test_idx), dr3)
    cv2.imwrite('./tmp/%s_%d_pred_paf.jpg' % (category, test_idx), dr4)
    cv2.imwrite('./tmp/%s_%d_pred_htall.jpg' % (category, test_idx), dr5)
    cv2.imwrite('./tmp/%s_%d_pred_kps.jpg' % (category, test_idx), dr6)
    cv2.imwrite('./tmp/%s_%d_ori_kps.jpg' % (category, test_idx), dr7)
    cv2.imshow('ori_heatmap', dr1)
    cv2.imshow('ori_paf', dr2)
    cv2.imshow('pred_heatmap', dr3)
    cv2.imshow('pred_paf', dr4)
    cv2.imshow('pred_htall', dr5)
    cv2.imshow('pred_kps', dr6)
    cv2.imshow('ori_kps', dr7)
    cv2.waitKey(0)
