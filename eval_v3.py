from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import pickle
import cv2
import mxnet as mx
import numpy as np
import pandas as pd

from dataset import FashionAIKPSDataSet
from model import load_model
from config import cfg
from utils import draw_heatmap, draw_kps
from utils import process_cv_img, detect_kps_v3, get_logger, mkdir


def calc_error(kps_pred, kps_gt, category):
    dist = lambda dx, dy: np.sqrt(np.square(dx) + np.square(dy))
    idx1, idx2 = cfg.EVAL_NORMAL_IDX[category]
    norm = dist(kps_gt[idx1, 0] - kps_gt[idx2, 0], kps_gt[idx1, 1] - kps_gt[idx2, 1])
    if norm == 0:
        return -1
    keep = kps_gt[:, 2] == 1
    kps_gt = kps_gt[keep]
    kps_pred = kps_pred[keep]
    error = dist(kps_pred[:, 0] - kps_gt[:, 0], kps_pred[:, 1] - kps_gt[:, 1])
    error[kps_pred[:, 2] == -1] = norm  # fill missing with norm, so error = 1
    error /= norm
    error = error.mean()
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--version', type=int, default=3)
    parser.add_argument('--use-cache', action='store_true')
    args = parser.parse_args()
    print(args)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    data_dir = cfg.DATA_DIR
    show = args.show
    save = args.save
    use_cache = args.use_cache
    version = args.version
    logger = get_logger()
    # model
    net = load_model(args.model, version)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # data
    df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    #df = df.sort_values(by='image_category')
    testdata = FashionAIKPSDataSet(df, False)
    num = len(testdata)
    base_name = './result/val'
    mkdir(base_name)
    for c in cfg.CATEGORY:
        mkdir('%s/%s' % (base_name, c))
    result = {k: [] for k in cfg.CATEGORY}
    record = []
    for i in range(num):
        path = os.path.join(data_dir, 'train', testdata.img_lst[i])
        img = cv2.imread(path)
        category = testdata.category[i]
        kps_gt = testdata.kps[i]
        heatmap = net.predict(img, ctx)
        # detect kps
        kps_pred = detect_kps_v3(img, heatmap, None, category)
        # calc_error
        error = calc_error(kps_pred, kps_gt, category)
        record.append((path, kps_gt, kps_pred, error))
        if error != -1:
            result[category].append(error)
        if i % 100 == 99:
            logger.info('Eval %d samples', i + 1)
            sum_err, sum_num = 0, 0
            for k in result:
                if result[k]:
                    err = np.array(result[k])
                    logger.info('Average Error for %s: %f', k, err.mean())
                    sum_err += err.sum()
                    sum_num += len(err)
            logger.info('Average Error %f', sum_err / sum_num)

        if show:
            landmark_idx = cfg.LANDMARK_IDX[category]
            heatmap = heatmap[landmark_idx].max(axis=0)

            dr1 = draw_heatmap(img, heatmap)
            dr2 = draw_kps(img, kps_pred)
            dr3 = draw_kps(img, kps_gt)

            cv2.imshow('heatmap', dr1)
            cv2.imshow('kps_pred', dr2)
            cv2.imshow('kps_gt', dr3)
            key = cv2.waitKey(0)
            if key == 27:
                break

    logger.info('Total Eval %d samples', num)
    sum_err, sum_num = 0, 0
    for k in result:
        if len(result[k]) != 0:
            err = np.array(result[k])
            logger.info('Total Average Error for %s: %f', k, err.mean())
            sum_err += err.sum()
            sum_num += len(err)
    logger.info('Total Average Error %f', sum_err / sum_num)
    pickle.dump([result, record], open('./result/eval_val.pkl', 'wb'))


if __name__ == '__main__':
    main()
