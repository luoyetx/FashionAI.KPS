from __future__ import print_function, division

import argparse
import numpy as np
import pandas as pd

from lib.config import cfg


def calc_error(kps_pred, kps_gt, category):
    dist = lambda dx, dy: np.sqrt(np.square(dx) + np.square(dy))
    idx1, idx2 = cfg.EVAL_NORMAL_IDX[category]
    if kps_gt[idx1, 2] == -1 or kps_gt[idx2, 2] == -1:
        return -1
    norm = dist(kps_gt[idx1, 0] - kps_gt[idx2, 0], kps_gt[idx1, 1] - kps_gt[idx2, 1])
    keep = kps_gt[:, 2] == 1
    kps_gt = kps_gt[keep]
    kps_pred = kps_pred[keep]
    if len(kps_gt) == 0:
        # all occ
        return -1
    error = dist(kps_pred[:, 0] - kps_gt[:, 0], kps_pred[:, 1] - kps_gt[:, 1])
    error[kps_pred[:, 2] == -1] = norm  # fill missing with norm, so error = 1
    error = error.mean() / norm
    return error


def read_csv(path):
    df = pd.read_csv(path)
    # img path
    img_lst = df['image_id'].tolist()
    category = df['image_category'].tolist()
    # kps, (x, y, v) v -> (not exists -1, occur 0, normal 1)
    cols = df.columns[2:]
    kps = []
    for i in range(cfg.NUM_LANDMARK):
        for j in range(3):
            kps.append(df[cols[i]].apply(lambda x: int(x.split('_')[j])).as_matrix())
    kps = np.vstack(kps).T.reshape((len(img_lst), -1, 3)).astype(np.float)
    return kps, category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='./data/val.csv')
    parser.add_argument('--pred', type=str, default='./result/val_result.csv')
    args = parser.parse_args()
    print(args)
    kps_gt, category = read_csv(args.gt)
    kps_pred, _ = read_csv(args.pred)
    assert len(kps_gt) == len(kps_pred)

    num_category = len(cfg.CATEGORY)
    result = [[] for i in range(num_category)]
    for gt, pred, cate in zip(kps_gt, kps_pred, category):
        cate_idx = cfg.CATEGORY.index(cate)
        err = calc_error(pred, gt, cate)
        if err != -1:
            result[cate_idx].append(err)

    result = [np.array(_) for _ in result]
    for i in range(num_category):
        category = cfg.CATEGORY[i]
        err = result[i].mean()
        print('Average Error for %s: %f' % (category, err))
    result = np.hstack(result)
    err = result.mean()
    print('Total Average Error %f' % err)


if __name__ == '__main__':
    main()
