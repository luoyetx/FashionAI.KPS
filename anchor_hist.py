from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import cfg
from dataset import FashionAIDetDataSet


def main():
    np.random.seed(0)
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'train.csv'))
    detset = FashionAIDetDataSet(df)

    fn = './tmp/anchors.npy'
    if os.path.exists(fn):
        print('load from', fn)
        boxes = np.load(fn)
    else:
        print('calculate boxes')
        boxes = []
        for idx, pack in enumerate(detset):
            _, label = pack
            boxes.append(label[:4])
            if idx % 1000 == 999:
                print("Process", idx + 1)
        boxes = np.array(boxes)
        np.save(fn, boxes)

    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]

    plt.figure()
    plt.hist(ws, bins=100)
    plt.title('width')
    plt.figure()
    plt.hist(hs, bins=100)
    plt.title('height')
    plt.figure()
    keep = ws > 0
    hs = hs[keep]
    ws = ws[keep]
    plt.hist(hs / ws, bins=100)
    plt.title('h / w')
    plt.show()


if __name__ == '__main__':
    main()
