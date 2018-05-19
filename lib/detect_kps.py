from __future__ import print_function, division

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from lib.config import cfg

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from lib.heatmap import pickPeeks


def predict_missing_with_center(img, kps, category):
    # fill missing
    h, w = img.shape[:2]
    landmark_idx = cfg.LANDMARK_IDX[category]
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


def detect_kps_v1(img, heatmap, paf, category):
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap.transpose((1, 2, 0)), (w, h), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf.transpose((1, 2, 0)), (w, h), interpolation=cv2.INTER_CUBIC)
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_ldm = len(landmark_idx)
    sigma = 1
    thres1 = 0.1
    num_mid = 10
    thres2 = 0.1
    # peaks
    peaks = []
    heatmap = heatmap[:, :, landmark_idx]
    for i in range(num_ldm):
        ht_ori = heatmap[:, :, i]
        ht = gaussian_filter(ht_ori, sigma=sigma)
        #ht = cv2.GaussianBlur(ht_ori, (7, 7), 0)
        mask = np.zeros_like(ht)
        pickPeeks(ht, mask, thres1)
        peak = zip(np.nonzero(mask)[1], np.nonzero(mask)[0]) # note reverse
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
            score = paf[:, :, idx]
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    norm = max(norm, 1e-5)
                    vec = np.divide(vec, norm)
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=num_mid), np.linspace(candA[i][1], candB[j][1], num=num_mid))
                    score_mid = np.array([score[int(round(startend[k][1])), int(round(startend[k][0]))] for k in range(len(startend))])
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
    # missing
    kps = predict_missing_with_center(img, kps, category)
    return kps


def detect_kps_v2(img, heatmap, paf, category):
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap.transpose((1, 2, 0)), (w, h), interpolation=cv2.INTER_CUBIC)
    landmark_idx = cfg.LANDMARK_IDX[category]
    num_ldm = cfg.NUM_LANDMARK
    sigma = 1
    thres1 = 0.1
    # peaks
    kps = np.zeros((num_ldm, 3))
    kps[:] = -1
    for idx in landmark_idx:
        ht_ori = heatmap[:, :, idx]
        #ht = gaussian_filter(ht_ori, sigma=sigma)
        ht = cv2.GaussianBlur(ht_ori, (7, 7), 0)
        mask = np.zeros_like(ht)
        pickPeeks(ht, mask, thres1)
        peak = zip(np.nonzero(mask)[1], np.nonzero(mask)[0]) # note reverse
        peak = np.array([[x[0], x[1], ht_ori[x[1], x[0]]] for x in peak])
        if len(peak) == 0:
            continue
        peak = peak[np.argsort(peak[:, 2])[::-1]]
        x, y, s = peak[0]
        kps[idx, 0] = x
        kps[idx, 1] = y
        kps[idx, 2] = 1
    # missing
    kps = predict_missing_with_center(img, kps, category)
    return kps


detect_kps = detect_kps_v2
