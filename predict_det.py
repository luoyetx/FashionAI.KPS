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
from lib.model import DetNet, load_model, multi_scale_predict, multi_scale_detection
from lib.utils import draw_heatmap, draw_paf, draw_kps, get_logger, crop_patch, draw_box
from lib.detect_kps import detect_kps_v1, detect_kps_v3
from lib.rpn import AnchorProposal


file_pattern = './result/tmp_%s_result_%d.csv'


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


def work_func(df, idx, args):
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = args.data_dir
    version = args.version
    show = args.show
    multi_scale = args.multi_scale
    logger = get_logger()
    # model
    feat_stride = cfg.FEAT_STRIDE
    scales = cfg.DET_SCALES
    ratios = cfg.DET_RATIOS
    anchor_proposals = [AnchorProposal(scales[i], ratios, feat_stride[i]) for i in range(2)]
    detnet = DetNet(anchor_proposals)
    creator, featname, fixed = cfg.BACKBONE_Det['resnet50']
    detnet.init_backbone(creator, featname, fixed, pretrained=False)
    detnet.load_params(args.det_model, ctx)
    detnet.hybridize()
    kpsnet = load_model(args.kps_model, version=version)
    kpsnet.collect_params().reset_ctx(ctx)
    kpsnet.hybridize()
    # data
    image_ids = df['image_id'].tolist()
    image_paths = [os.path.join(data_dir, img_id) for img_id in image_ids]
    image_categories = df['image_category'].tolist()
    # run
    result = []
    for i, (path, category) in enumerate(zip(image_paths, image_categories)):
        img = cv2.imread(path)
        # detection
        h, w = img.shape[:2]
        dets = multi_scale_detection(detnet, ctx, img, category)
        if len(dets) != 0:
            bbox = dets[0, :4]
            score = dets[0, -1]
        else:
            bbox = [0, 0, w, h]
            score = 0
        bbox = get_border(bbox, w, h, 0.2)
        roi = crop_patch(img, bbox)
        # predict kps
        if version == 2:
            heatmap, paf = multi_scale_predict(kpsnet, ctx, version, roi, category, multi_scale)
            kps_pred = detect_kps_v1(roi, heatmap, paf, category)
        elif version == 3:
            heatmap = multi_scale_predict(kpsnet, ctx, version, roi, category, multi_scale)
            kps_pred = detect_kps_v3(roi, heatmap, category)
        elif version == 4:
            pass
        elif version == 5:
            heatmap = multi_scale_predict(kpsnet, ctx, version, roi, category, multi_scale)
            kps_pred = detect_kps_v3(roi, heatmap, category)
        else:
            raise RuntimeError('no such version %d'%version)
        x1, y1 = bbox[:2]
        kps_pred[:, 0] += x1
        kps_pred[:, 1] += y1
        result.append(kps_pred)
        # show
        if show:
            landmark_idx = cfg.LANDMARK_IDX[category]
            dr0 = draw_box(img, bbox, '%s_%.2f' % (category, score))
            cv2.imshow('det', dr0)
            if version == 2:
                htall = heatmap[-1]
                heatmap = heatmap[landmark_idx].max(axis=0)
                dr1 = draw_heatmap(roi, heatmap)
                dr2 = draw_paf(roi, paf)
                dr3 = draw_kps(img, kps_pred)
                dr4 = draw_heatmap(roi, htall)
                cv2.imshow('heatmap', dr1)
                cv2.imshow('paf', dr2)
                cv2.imshow('kps_pred', dr3)
                cv2.imshow('htall', dr4)
            elif version == 3:
                heatmap = heatmap[landmark_idx].max(axis=0)
                dr1 = draw_heatmap(roi, heatmap)
                dr2 = draw_kps(img, kps_pred)
                cv2.imshow('heatmap', dr1)
                cv2.imshow('kps_pred', dr2)
            elif version == 4:
                pass
            elif version == 5:
                heatmap = heatmap[landmark_idx].max(axis=0)
                dr1 = draw_heatmap(roi, heatmap)
                dr2 = draw_kps(img, kps_pred)
                cv2.imshow('heatmap', dr1)
                cv2.imshow('kps_pred', dr2)
            else:
                raise RuntimeError('no such version %d'%version)
            key = cv2.waitKey(0)
            if key == 27:
                break
        if i % 100 == 0:
            logger.info('Worker %d process %d samples', idx, i + 1)
    # save
    fn = file_pattern % (args.type, idx)
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
    parser.add_argument('--det-model', type=str, required=True)
    parser.add_argument('--kps-model', type=str, required=True)
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--type', type=str, default='val', choices=['val', 'test'])
    args = parser.parse_args()
    print(args)
    # data
    if args.type == 'val':
        data_dir = cfg.DATA_DIR
        df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    else:
        data_dir = os.path.join(cfg.DATA_DIR, 'r2-test-a')
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
    result = pd.concat([pd.read_csv(file_pattern % (args.type, i)) for i in range(num_worker)])
    result.to_csv('./result/%s_result.csv' % args.type, index=False)


if __name__ == '__main__':
    main()
