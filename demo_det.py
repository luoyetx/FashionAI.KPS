from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import cv2
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import pandas as pd
import seaborn as sns

from lib.dataset import FashionAIDetDataSet
from lib.model import load_model, DetNet
from lib.rpn import AnchorProposal
from lib.utils import draw_det, draw_box, reverse_to_cv_img, crop_patch, draw_kps, draw_heatmap
from lib.detect_kps import detect_kps_v1, detect_kps_v3
from lib.config import cfg


def get_border(bbox, w, h, expand=0.1):
    xmin, ymin, xmax, ymax = bbox
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--det-model', type=str, required=True)
    parser.add_argument('--kps-model', type=str, required=True)
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
    feat_stride = cfg.FEAT_STRIDE
    scales = cfg.DET_SCALES
    ratios = cfg.DET_RATIOS
    anchor_proposal = AnchorProposal(scales, ratios, feat_stride)
    detnet = DetNet(anchor_proposal)
    creator, featname, fixed = cfg.BACKBONE_Det['resnet50']
    detnet.init_backbone(creator, featname, fixed, pretrained=False)
    detnet.load_params(args.det_model, ctx)
    detnet.hybridize()
    kpsnet = load_model(args.kps_model, 2)
    kpsnet.collect_params().reset_ctx(ctx)
    kpsnet.hybridize()
    # data
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    testdata = FashionAIDetDataSet(df_test, is_train=False)

    for test_idx in range(len(testdata)):
        category = df_test['image_category'][test_idx]
        landmark_idx = cfg.LANDMARK_IDX[category]

        data, rois = testdata[test_idx]
        img = reverse_to_cv_img(data)
        h, w = img.shape[:2]
        dets = detnet.predict(img, ctx)

        cate_idx = cfg.CATEGORY.index(category)
        bbox = dets[cate_idx][0, :4]
        score = dets[cate_idx][0, -1]
        bbox = get_border(bbox, w, h, 0.2)
        det_im = draw_box(img, bbox, '%s_%.2f' % (category, score))

        x1, y1, x2, y2 = [int(_) for _ in bbox]
        print(x1, y1, x2, y2, x2 - x1, y2 - y1)
        cv2.imshow('dets', det_im)

        roi = crop_patch(img, bbox)
        heatmap, paf = kpsnet.predict(roi, ctx)
        kps_pred = detect_kps_v1(roi, heatmap, paf, category)

        ht = heatmap[landmark_idx].max(axis=0)
        dr1 = draw_heatmap(roi, ht)
        dr2 = draw_kps(roi, kps_pred)
        cv2.imshow('ht', dr1)
        cv2.imshow('kps', dr2)

        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    main()
