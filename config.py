from easydict import EasyDict
import numpy as np
from mxnet import gluon as gl

cfg = EasyDict()

cfg.DATA_DIR = './data'
cfg.TRAIN_RATE = 0.9

cfg.CATEGORY = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
cfg.NUM_LANDMARK = 24
cfg.LANDMARK_IDX = {
    'blouse': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
    'skirt': [15, 16, 17, 18],
    'outwear': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'dress': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
    'trousers': [15, 16, 19, 20, 21, 22, 23],
}
cfg.LANDMARK_SWAP = [(0, 1), (3, 4), (5, 6), (10, 12), (9, 11), (7, 8), (13, 14), (15, 16), (17, 18), (21, 23), (20, 22),]
cfg.PAF_LANDMARK_PAIR = [(0, 1), (0, 2), (0, 3),
                         (1, 2), (1, 4),
                         (2, 5), (2, 6),
                         (3, 5), (3, 10),
                         (4, 6), (4, 12),
                         (5, 6), (5, 7), (5, 9), (5, 13),
                         (6, 8), (6, 11), (6, 14),
                         (7, 8), (7, 13),
                         (8, 14),
                         (9, 10),
                         (11, 12),
                         (13, 14),
                         (15, 16), (15, 17), (15, 19), (15, 21),
                         (16, 18), (16, 19), (16, 23),
                         (17, 18),
                         (19, 20), (19, 22),
                         (20, 21),
                         (22, 23),]

cfg.HEATMAP_THRES = 1
cfg.CROP_SIZE = 368
cfg.ROT_MAX = 40
cfg.SCALE_MIN_RATE = 0.6
cfg.SCALE_MAX_RATE = 1.2
cfg.CROP_CENTER_OFFSET_MAX = 40

cfg.PIXEL_MEAN = [0.485, 0.456, 0.406]  # RGB
cfg.PIXEL_STD = [0.229, 0.224, 0.225]

cfg.BACKBONE_v2 = {
    'vgg16': (gl.model_zoo.vision.vgg16, 'relu8_fwd', ['conv0', 'conv1', 'conv2', 'conv3']),
    'vgg19': (gl.model_zoo.vision.vgg19, 'relu9_fwd', ['conv0', 'conv1', 'conv2', 'conv3']),
    'resnet50': (gl.model_zoo.vision.resnet50_v2, 'stage2__plus3', ['conv0']),
}

cfg.BACKBONE_v3 = {
    'resnet50': (gl.model_zoo.vision.resnet50_v2, ['stage1__plus2', 'stage2__plus3', 'stage3__plus5'], ['conv0']),
}

cfg.BACKBONE_v4 = {
    'vgg19': (gl.model_zoo.vision.vgg19, 'relu9_fwd', ['conv0', 'conv1', 'conv2', 'conv3']),
}

cfg.EVAL_NORMAL_IDX = {
    'blouse': (5, 6),
    'skirt': (15, 16),
    'outwear': (5, 6),
    'dress': (5, 6),
    'trousers': (15, 16),
}

cfg.SCALE_MIN = cfg.CROP_SIZE * cfg.SCALE_MIN_RATE
cfg.SCALE_MAX = cfg.CROP_SIZE * cfg.SCALE_MAX_RATE
cfg.PIXEL_MEAN = np.array(cfg.PIXEL_MEAN, dtype='float32').reshape((3, 1, 1))
cfg.PIXEL_STD = np.array(cfg.PIXEL_STD, dtype='float32').reshape((3, 1, 1))
cfg.FILL_VALUE = [int(v*255) for v in cfg.PIXEL_MEAN.flatten()[::-1]]  # BGR for opencv
