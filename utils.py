from __future__ import print_function, division

import os
import logging
import cv2
import mxnet as mx
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from model import PoseNet
from config import cfg


def process_cv_img(img):
    # HWC -> CHW, BGR -> RGB
    img = img.astype('float32').transpose((2, 0, 1)) / 255
    img = img[::-1, :, :]
    img = (img - cfg.PIXEL_MEAN) / cfg.PIXEL_STD
    return img

def reverse_to_cv_img(data):
    img = ((data * cfg.PIXEL_STD + cfg.PIXEL_MEAN) * 255).astype('uint8')
    img = img.transpose((1, 2, 0))[:, :, ::-1]
    return img


def crop_patch(img, bbox, fill_value=cfg.FILL_VALUE):
    height, width = img.shape[:-1]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
        print('[WARN] ridiculous x1, y1, x2, y2')
        return None
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        # out of boundary, still crop the face
        h, w = y2 - y1, x2 - x1
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        patch[:, :] = fill_value
        vx1 = 0 if x1 < 0 else x1
        vy1 = 0 if y1 < 0 else y1
        vx2 = width if x2 > width else x2
        vy2 = height if y2 > height else y2
        sx = -x1 if x1 < 0 else 0
        sy = -y1 if y1 < 0 else 0
        vw = vx2 - vx1
        vh = vy2 - vy1
        patch[sy:sy+vh, sx:sx+vw] = img[vy1:vy2, vx1:vx2]
        return patch
    return img[y1:y2, x1:x2]


def draw_kps(im, kps):
    im = im.copy()
    num_kps = len(kps)
    palette = np.array(sns.color_palette("hls", num_kps))
    palette = (palette * 255).astype('uint8')[:, ::-1].tolist()
    for idx, (x, y, v) in enumerate(kps):
        x, y = int(x), int(y)
        if v == 0:
            color = (0, 0, 0)
        elif v == 1:
            color = palette[idx]
        if v != -1:
            cv2.circle(im, (x, y), 3, color, -1)
            cv2.putText(im, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    return im


def draw_heatmap(im, ht):
    assert len(ht.shape) == 2, ht.shape
    h, w = im.shape[:2]
    ht = cv2.resize(ht, (w, h))
    ht[ht < 0] = 0
    ht[ht > 1] = 1
    ht = (ht * 255).astype(np.uint8)
    ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
    drawed = cv2.addWeighted(im, 0.5, ht, 0.5, 0)
    return drawed


def draw_paf(im, paf):
    assert len(paf.shape) == 3, paf.shape
    n, h, w = paf.shape
    paf = paf.reshape((n // 2, 2, h, w))
    paf = np.sqrt(np.square(paf[:, 0]) + np.square(paf[:, 1]))
    paf = paf.max(axis=0)
    return draw_heatmap(im, paf)


def parse_from_name(name):
    # name = /path/to/vgg16-S5-C64-BS16-adam-0100.params
    name = os.path.basename(name)
    name = name.split('.')[0]
    ps = name.split('-')
    backbone = ps[0]
    stages = int(ps[1][1:])
    channels = int(ps[2][1:])
    return backbone, stages, channels

def get_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def load_model(model):
    num_kps = cfg.NUM_LANDMARK
    num_limb = len(cfg.PAF_LANDMARK_PAIR)
    backbone, cpm_stages, cpm_channels = parse_from_name(model)
    net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
    creator, featname, fixed = cfg.BACKBONE[backbone]
    net.init_backbone(creator, featname, fixed)
    net.load_params(model, mx.cpu(0))
    return net


def detect_kps(img, heatmap, paf, category):
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap.transpose((1, 2, 0)), (w, h))
    paf = cv2.resize(paf.transpose((1, 2, 0)), (w, h))
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_ldm = len(landmark_idx)
    sigma = 1
    thres1 = 0.1
    num_mid = 10
    thres2 = 0.05
    # peaks
    peaks = []
    heatmap = heatmap[:, :, landmark_idx]
    for i in range(num_ldm):
        ht_ori = heatmap[: , :, i]
        ht = gaussian_filter(ht_ori, sigma=sigma)
        ht_left = np.zeros(ht.shape)
        ht_left[1:,:] = ht[:-1,:]
        ht_right = np.zeros(ht.shape)
        ht_right[:-1,:] = ht[1:,:]
        ht_up = np.zeros(ht.shape)
        ht_up[:,1:] = ht[:,:-1]
        ht_down = np.zeros(ht.shape)
        ht_down[:,:-1] = ht[:,1:]
        peak_binary = np.logical_and.reduce((ht>ht_left, ht>ht_right, ht>ht_up, ht>ht_down, ht > thres1))
        peak = zip(np.nonzero(peak_binary)[1], np.nonzero(peak_binary)[0]) # note reverse
        peak_with_score_links = [[x[0], x[1], ht_ori[x[1], x[0]], 0, 0] for x in peak]
        peaks.append(peak_with_score_links)
    # connection
    for idx, (ldm1, ldm2) in enumerate(cfg.PAF_LANDMARK_PAIR):
        if ldm1 not in landmark_idx or ldm2 not in landmark_idx:
            continue
        ldm1 = landmark_idx.index(ldm1)
        ldm2 = landmark_idx.index(ldm2)
        candA = peaks[ldm1]
        candB = peaks[ldm2]
        nA = len(candA)
        nB = len(candB)
        if nA != 0 and nB != 0:
            connection_candidate = []
            score = paf[:, :, 2*idx: 2*idx+2]
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    norm = max(norm, 1e-5)
                    vec = np.divide(vec, norm)
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=num_mid), np.linspace(candA[i][1], candB[j][1], num=num_mid))
                    vec_x = np.array([score[int(round(startend[k][1])), int(round(startend[k][0])), 0] for k in range(len(startend))])
                    vec_y = np.array([score[int(round(startend[k][1])), int(round(startend[k][0])), 1] for k in range(len(startend))])
                    score_mid = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_mid) / len(score_mid) + min(0.5*h/norm - 1, 0)
                    c1 = (score_mid > thres2).sum() > 0.8 * len(score_mid)
                    c2 = score_with_dist_prior > 0
                    if c1 and c2:
                        connection_candidate.append([i, j, score_with_dist_prior])
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 3))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c]
                if i not in connection[:, 0] and j not in connection[:, 1]:
                    connection = np.vstack([connection, [i, j, s]])
                    if len(connection) >= min(nA, nB):
                        break
            for i, j, s in connection:
                i, j = int(i), int(j)
                candA[i][3] += 1
                candB[j][3] += 1
                candA[i][4] += s
                candB[j][4] += s
    # detect kps
    num_kps = cfg.NUM_LANDMARK
    kps = np.zeros((num_kps, 3), dtype='int32')
    kps[:, :] = -1
    for i in range(num_ldm):
        cand = peaks[i]
        if len(cand) >= 1:
            idx = 0
            max_links = cand[0][3]
            max_score = cand[0][2]
            for j in range(1, len(cand)):
                if cand[j][3] > max_links or (cand[j][3] == max_links and cand[j][2] > max_score):
                    max_links = cand[j][3]
                    max_score = cand[j][2]
                    idx = j
            # if len(cand) > 1:
            #     print(i, 'select with', cand[idx][3], 'links and', cand[idx][2], 'score')
            j = landmark_idx[i]
            kps[j, 0] = cand[idx][0]
            kps[j, 1] = cand[idx][1]
            kps[j, 2] = 1
    # cheat
    keep = kps[:, 2] == 1
    if keep.sum() != 0:
        xmin = kps[keep, 0].min()
        xmax = kps[keep, 0].max()
        ymin = kps[keep, 1].min()
        ymax = kps[keep, 1].max()
        xc = (xmin + xmax) // 2
        yc = (ymin + ymax) // 2
    else:
        xc = w // 2
        yc = h // 2
    for idx in landmark_idx:
        if kps[idx, 2] == -1:
            kps[idx, 0] = xc
            kps[idx, 1] = yc
            kps[idx, 2] = 0
    return kps
