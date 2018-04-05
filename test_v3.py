from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import time
import argparse
import cv2
import mxnet as mx
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from model import load_model
from config import cfg
from utils import draw_heatmap, draw_kps
from utils import detect_kps_v3, process_cv_img, get_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--version', type=int, default=3)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    print(args)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = cfg.DATA_DIR
    version = args.version
    show = args.show
    logger = get_logger()
    # model
    net = load_model(args.model, version)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # data
    df = pd.read_csv(os.path.join(data_dir, 'test/test.csv'))
    result = []
    for i, row in df.iterrows():
        img_id = row['image_id']
        category = row['image_category']
        path = os.path.join(data_dir, 'test', row['image_id'])
        img = cv2.imread(path)
        # predict
        heatmap = net.predict(img, ctx)
        # detect kps
        kps_pred = detect_kps_v3(img, heatmap, None, category)
        result.append((img_id, category, kps_pred))
        if i % 100 == 0:
            logger.info('Process %d samples', i + 1)

        if show:
            landmark_idx = cfg.LANDMARK_IDX[category]
            heatmap = heatmap[landmark_idx].max(axis=0)

            dr1 = draw_heatmap(img, heatmap)
            dr3 = draw_kps(img, kps_pred)

            cv2.imshow('heatmap', dr1)
            cv2.imshow('detect', dr3)
            key = cv2.waitKey(0)
            if key == 27:
                break

    with open('./result/result.csv', 'w') as fout:
        header = 'image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out\n'
        fout.write(header)
        for img_id, category, kps in result:
            fout.write(img_id)
            fout.write(',%s'%category)
            for p in kps:
                s=',%d_%d_%d'%(p[0], p[1], p[2])
                fout.write(s)
            fout.write('\n')
