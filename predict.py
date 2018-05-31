from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import multiprocessing as mp
import argparse
import cv2
import mxnet as mx
import numpy as np
import pandas as pd

from lib.config import cfg
from lib.model import load_model, multi_scale_predict
from lib.utils import draw_heatmap, draw_paf, draw_kps, get_logger, crop_patch
from lib.detect_kps import detect_kps
from lib.dataset import get_border


file_pattern = './result/%s_%s_result_%d.csv'


def work_func(df, idx, args):
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = args.data_dir
    model_path = args.model
    version = args.version
    scale = args.scale
    show = args.show
    multi_scale = args.multi_scale
    logger = get_logger()
    # model
    net = load_model(model_path, version=version, scale=scale)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    if args.emodel != '':
        enet = load_model(args.emodel, version=args.eversion, scale=scale)
        enet.collect_params().reset_ctx(ctx)
        enet.hybridize()
    else:
        enet = None
    # data
    image_ids = df['image_id'].tolist()
    image_paths = [os.path.join(data_dir, img_id) for img_id in image_ids]
    image_categories = df['image_category'].tolist()
    # run
    result = []
    for i, (path, category) in enumerate(zip(image_paths, image_categories)):
        img = cv2.imread(path)
        # predict
        heatmap, paf = multi_scale_predict(net, ctx, img, multi_scale)
        if enet:
            eheatmap, epaf = multi_scale_predict(enet, ctx, img, multi_scale)
            heatmap = (heatmap + eheatmap) / 2
            paf = (paf + epaf) / 2
        kps_pred = detect_kps(img, heatmap, paf, category)
        result.append(kps_pred)
        # show
        if show:
            landmark_idx = cfg.LANDMARK_IDX[category]
            ht = cv2.GaussianBlur(heatmap, (7, 7), 0)
            ht = ht[landmark_idx].max(axis=0)
            heatmap = heatmap[landmark_idx].max(axis=0)
            cv2.imshow('heatmap', draw_heatmap(img, heatmap))
            cv2.imshow('heatmap_blur', draw_heatmap(img, ht))
            cv2.imshow('kps_pred', draw_kps(img, kps_pred))
            cv2.imshow('paf', draw_paf(img, paf))
            key = cv2.waitKey(0)
            if key == 27:
                break
        if i % 100 == 0:
            logger.info('Worker %d process %d samples', idx, i + 1)
    # save
    fn = file_pattern % (args.prefix, args.type, idx)
    with open(fn, 'w') as fout:
        header = 'image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out\n'
        fout.write(header)
        for img_id, category, kps in zip(image_ids, image_categories, result):
            fout.write(img_id)
            fout.write(',%s'%category)
            for p in kps:
                s = ',%d_%d_%d' % (p[0], p[1], p[2])
                fout.write(s)
            fout.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--version', type=int, default=3)
    parser.add_argument('--scale', type=int, default=0)
    parser.add_argument('--emodel', type=str, default='')
    parser.add_argument('--eversion', type=int, default=4)
    parser.add_argument('--escale', type=int, default=0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--type', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--prefix', type=str, default='tmp')
    args = parser.parse_args()
    print(args)
    # data
    if args.type == 'val':
        data_dir = cfg.DATA_DIR
        df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    else:
        data_dir = os.path.join(cfg.DATA_DIR, 'r2-test-b')
        df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    args.data_dir = data_dir
    #df = df.sample(frac=1)
    num_worker = args.num_worker
    num_sample = len(df) // num_worker + 1
    dfs = [df[i*num_sample: (i+1)*num_sample] for i in range(num_worker)]
    # run
    workers = [mp.Process(target=work_func, args=(dfs[i], i, args)) for i in range(num_worker)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    # merge
    result = pd.concat([pd.read_csv(file_pattern % (args.prefix, args.type, i)) for i in range(num_worker)])
    result.to_csv('./result/%s_%s_result.csv' % (args.prefix, args.type), index=False)


if __name__ == '__main__':
    main()
