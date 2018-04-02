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

from model import PoseNet
from config import cfg
from utils import draw_heatmap, draw_paf, draw_kps
from utils import detect_kps, process_cv_img, get_logger, load_model, mkdir


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
    df = pd.read_csv(os.path.join(data_dir, 'test/test.csv'))
    base_name = './result/test'
    mkdir(base_name)
    for c in cfg.CATEGORY:
        mkdir('%s/%s' % (base_name, c))
    result = []
    for i, row in df.iterrows():
        img_id = row['image_id']
        category = row['image_category']
        path = os.path.join(data_dir, 'test', row['image_id'])
        img = cv2.imread(path)
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
        result.append((img_id, category, kps_pred))
        if i % 100 == 0:
            logger.info('Process %d samples', i + 1)

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
