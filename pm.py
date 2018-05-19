from __future__ import print_function

import argparse
import pandas as pd

from lib.config import cfg
from lib.dataset import FashionAIKPSDataSet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./result/tmp_test_result.csv')
    args = parser.parse_args()
    print(args)

    test = pd.read_csv(args.csv)
    test = FashionAIKPSDataSet(test)

    c = 0
    for i in range(len(test)):
        cate = test.category[i]
        kps = test.kps[i]
        landmark_idx = cfg.LANDMARK_IDX[cate]
        for j in landmark_idx:
            if kps[j, 2] == 0:
                c += 1
    print('missing points', c)


if __name__ == '__main__':
    main()
