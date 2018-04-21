from __future__ import print_function

import os
import pickle
import cv2
import pandas as pd
import imagehash
from PIL import Image


def hashit(df, root):
    img_lst = df['image_id'].tolist()
    hash = []
    for i in range(len(img_lst)):
        path = os.path.join(root, img_lst[i])
        h = imagehash.phash(Image.open(path))
        hash.append((path, h))
        if i % 100 == 99:
            print('Process', i + 1)
    return hash


def hashall():
    print('Process train')
    train = pd.read_csv('./data/train/Annotations/train.csv')
    hash = hashit(train, './data/train')
    pickle.dump(hash, open('./data/train_hash.pkl', 'wb'))
    print('Process warmup')
    warmup = pd.read_csv('./data/train/Annotations/annotations.csv')
    hash = hashit(warmup, './data/train')
    pickle.dump(hash, open('./data/warmup_hash.pkl', 'wb'))
    print('Process test-a')
    testa = pd.read_csv('./data/test/test.csv')
    hash = hashit(testa, './data/test')
    pickle.dump(hash, open('./data/testa_hash.pkl', 'wb'))
    print('Process test-b')
    testb = pd.read_csv('./data/test-b/test.csv')
    hash = hashit(testb, './data/test-b')
    pickle.dump(hash, open('./data/testb_hash.pkl', 'wb'))


def searchit(db, df, merge=False, show=False):
    c = 0
    for img_path, h in df:
        if h in db:
            print(img_path, '-- dups with --', db[h])
            if show:
                ima = cv2.imread(img_path)
                cv2.imshow('db', ima)
                imb = cv2.imread(db[h])
                cv2.imshow('df', imb)
                k = cv2.waitKey(0)
                if k == 27:
                    break
            c +=1
        elif merge:
            db[h] = img_path
    print('dups number:', c)


def search():
    # train
    print('search train')
    train = pickle.load(open('./data/train_hash.pkl', 'rb'))
    db = {}
    searchit(db, train, True)
    # warmup
    print('search warmup')
    warmup = pickle.load(open('./data/warmup_hash.pkl', 'rb'))
    searchit(db, warmup, True)
    # test-a
    print('search test-a')
    testa = pickle.load(open('./data/testa_hash.pkl', 'rb'))
    searchit(db, testa, False)
    # test-b
    print('search test-b')
    testb = pickle.load(open('./data/testb_hash.pkl', 'rb'))
    searchit(db, testb, False, True)


if __name__ == '__main__':
    #hashall()
    search()
