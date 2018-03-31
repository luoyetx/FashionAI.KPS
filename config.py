from easydict import EasyDict
import numpy as np
from mxnet import gluon as gl

cfg = EasyDict()

cfg.IMG_DIR = './data'

cfg.CATEGORY_2_IDX = {
    'blouse': 0,
    'skirt': 1,
    'outwear': 2,
    'dress': 3,
    'trousers': 4
}
cfg.IDX_2_CATEGORY = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']

cfg.LANDMARK_IDX = {
    'blouse': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
    'skirt': [15, 16, 17, 18],
    'outwear': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'dress': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
    'trousers': [15, 16, 19, 20, 21, 22, 23],
}
cfg.LANDMARK_SWAP = {
    'blouse': [(0, 1), (3, 4), (5, 6), (7, 9), (8, 10), (11, 12)],
    'skirt': [(0, 1), (2, 3)],
    'outwear': [(0, 1), (2, 3), (4, 5), (6, 7), (9, 11), (8, 10), (12, 13)],
    'dress': [(0, 1), (3, 4), (5, 6), (7, 8), (10, 12), (9, 11), (13, 14)],
    'trousers': [(0, 1), (4, 6), (3, 5)],
}
cfg.PAF_LANDMARK_PAIR = {
    'blouse': [(2, 0), (2, 1), (0, 3), (3, 8), (8, 7), (7, 5), (5, 11), (1, 4), (4, 10), (10, 9), (9, 6), (6, 12)],
    'skirt': [(0, 1), (0, 2), (1, 3)],
    'outwear': [(0, 1), (1, 3), (3, 11), (11, 10), (10, 5), (5, 7), (5, 13), (7, 13), (0, 2), (2, 9), (9, 8), (8, 4), (4, 6), (4, 12), (6, 12)],
    'dress': [(2, 0), (2, 1), (0, 3), (3, 10), (10, 9), (9, 5), (5, 7), (5, 13), (7, 13), (1, 4), (4, 12), (12, 11), (11, 6), (6, 8), (6, 14), (8, 14)],
    'trousers': [(2, 0), (2, 1), (2, 3), (2, 5), (0, 4), (1, 6)],
}

cfg.SIGMA = 7
cfg.THRE = 1
cfg.STRIDE = 8
cfg.CROP_SIZE = 368
cfg.ROT_MAX = 20
cfg.SCALE_MIN_RATE = 0.6
cfg.SCALE_MAX_RATE = 1.1
cfg.CROP_CENTER_OFFSET_MAX = 40

cfg.PIXEL_MEAN = [0.485, 0.456, 0.406]
cfg.PIXEL_STD = [0.229, 0.224, 0.225]

cfg.BACKBONE = {
    'vgg16': (gl.model_zoo.vision.vgg16, 'relu8', ['conv0', 'conv1', 'conv2', 'conv3']),
    'vgg19': (gl.model_zoo.vision.vgg19, 'relu9', ['conv0', 'conv1', 'conv2', 'conv3']),
    'reset50': (gl.model_zoo.vision.resnet50_v2, '', []),
}

cfg.TEST_IMAGE = {
    'blouse': [],
    'skirt': 'train/Images/skirt/0010590c4110a37f76f109d079efd8ca.jpg',
    'outwear': [],
    'dress': [],
    'trousers': []
}

cfg.SCALE_MIN = cfg.CROP_SIZE * cfg.SCALE_MIN_RATE
cfg.SCALE_MAX = cfg.CROP_SIZE * cfg.SCALE_MAX_RATE
