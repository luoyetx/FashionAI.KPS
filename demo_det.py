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

from dataset import FashionAIDetDataSet
from model import load_model, DetNet
from rpn import AnchorProposal
from utils import draw_heatmap, draw_paf, draw_kps, reverse_to_cv_img
from detect_kps import detect_kps_v1, detect_kps_v3
from config import cfg


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
    scales = [5, 10, 20]
    ratios = [1, 0.5, 2]
    anchor_proposal = AnchorProposal(scales, ratios, 16)
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

        num_category = 5
        palette = np.array(sns.color_palette("hls", num_category))
        palette = (palette * 255).astype('uint8')[:, ::-1].tolist()
        for i in range(num_category):
            category = cfg.CATEGORY[i]
            color = palette[i]
            for proposal, score in zip(*dets[i]):
                x1, y1, x2, y2 = [int(_) for _ in proposal]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, '%s-%0.2f' % (category, score), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.imshow('dets', img)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    main()
