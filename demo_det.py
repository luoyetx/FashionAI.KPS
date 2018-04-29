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
from lib.utils import draw_det, reverse_to_cv_img
from lib.detect_kps import detect_kps_v1, detect_kps_v3
from lib.config import cfg


def main():
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
    feat_stride = cfg.FEAT_STRIDE
    scales = cfg.DET_SCALES
    ratios = cfg.DET_RATIOS
    anchor_proposal = AnchorProposal(scales, ratios, feat_stride)
    net = DetNet(anchor_proposal.num_anchors)
    creator, featname, fixed = cfg.BACKBONE_Det['resnet50']
    net.init_backbone(creator, featname, fixed, pretrained=False)
    net.load_params(args.model, ctx)
    net.hybridize()
    # data
    df_test = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    testdata = FashionAIDetDataSet(df_test, is_train=False)

    for test_idx in range(len(testdata)):
        category = df_test['image_category'][test_idx]
        landmark_idx = cfg.LANDMARK_IDX[category]

        data, rois = testdata[test_idx]
        img = reverse_to_cv_img(data)
        dets = net.predict(img, ctx, anchor_proposal)
        assert len(dets) == 1
        dets = dets[0]

        img = draw_det(img, dets, category)
        cv2.imshow('dets', img)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    main()
