from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import cv2
import mxnet as mx
import numpy as np
import pandas as pd

from model import load_model
from config import cfg
from utils import draw_heatmap, draw_paf, draw_kps
from utils import detect_kps_v1, detect_kps_v3, get_logger


def predict(img_path, category, model_path, ctx, version, show=False):
    net = get_model(model_path, version, ctx)

    return kps_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    print(args)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = cfg.DATA_DIR
    model_path = args.model
    version = args.version
    show = args.show
    logger = get_logger()
    # model
    net = load_model(model_path, version=version)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # data
    df = pd.read_csv(os.path.join(data_dir, 'test/test.csv'))
    image_ids = df['image_id'].tolist()
    image_paths = [os.path.join(data_dir, 'test', img_id) for img_id in image_ids]
    image_categories = df['image_category'].tolist()
    # run
    result = []
    for i, (path, category) in enumerate(zip(image_paths, image_categories)):
        img = cv2.imread(path)
        # predict
        if version == 2:
            heatmap, paf = net.predict(img, ctx)
            kps_pred = detect_kps_v1(img, heatmap, paf, category)
        else:
            heatmap = net.predict(img, ctx)
            kps_pred = detect_kps_v3(img, heatmap, category)
        result.append(kps_pred)
        # show
        if show:
            landmark_idx = cfg.LANDMARK_IDX[category]
            if version == 2:
                htall = heatmap[-1]
                heatmap = heatmap[landmark_idx].max(axis=0)
                dr1 = draw_heatmap(img, heatmap)
                dr2 = draw_paf(img, paf)
                dr3 = draw_kps(img, kps_pred)
                dr4 = draw_heatmap(img, htall)
                cv2.imshow('heatmap', dr1)
                cv2.imshow('paf', dr2)
                cv2.imshow('kps_pred', dr3)
                cv2.imshow('htall', dr4)
            else:
                heatmap = heatmap[landmark_idx].max(axis=0)
                dr1 = draw_heatmap(img, heatmap)
                dr2 = draw_kps(img, kps_pred)
                cv2.imshow('heatmap', dr1)
                cv2.imshow('kps_pred', dr2)
            key = cv2.waitKey(0)
            if key == 27:
                break
        if i % 100 == 0:
            logger.info('Process %d samples', i + 1)

    # save
    with open('./result/result.csv', 'w') as fout:
        header = 'image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out\n'
        fout.write(header)
        for img_id, category, kps in zip(image_ids, image_categories, result):
            fout.write(img_id)
            fout.write(',%s'%category)
            for p in kps:
                s=',%d_%d_%d' % (p[0], p[1], p[2])
                fout.write(s)
            fout.write('\n')


if __name__ == '__main__':
    main()
