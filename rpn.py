# coding=utf-8

from __future__ import print_function, division

import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn

from config import cfg
from utils import process_cv_img
from generate_anchors import generate_anchors
from bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from bbox import bbox_overlaps_cython
from cpu_nms import cpu_nms


class AnchorProposal(object):

    def __init__(self, scales, ratios, feat_stride):
        super(AnchorProposal, self).__init__()
        scales = np.array(scales)
        ratios = np.array(ratios)
        self.feat_stride = feat_stride
        self.anchors = generate_anchors(feat_stride, ratios, scales)
        self.num_anchors = self.anchors.shape[0]
        # train
        self.allowed_border = 0
        self.anchor_negative_overlap = 0.3
        self.anchor_positive_overlap = 0.5
        self.anchor_fg_fraction = 0.25
        self.anchor_per_sample = 256
        self.hard_mining = True
        # test
        self.num_det_per_category = 300
        self.min_size = 0
        self.nms_th = 0.3

    def target(self, rpn_cls, gt_boxes, im_info=(368, 368)):
        ctx = rpn_cls.context
        n, _, height, width = rpn_cls.shape
        num_category = 5
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.shape[0]
        all_anchors = (self.anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self.allowed_border) &
            (all_anchors[:, 1] >= -self.allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self.allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self.allowed_border)    # height
        )[0]

        # keep only inside anchors
        if inds_inside.shape[0] == 0:
            # If no anchors inside use whatever anchors we have
            inds_inside = np.arange(0, all_anchors.shape[0])

        anchors = all_anchors[inds_inside, :]

        rpn_cls = rpn_cls.asnumpy()
        gt_boxes = gt_boxes.asnumpy()
        batch_labels = np.zeros((n, num_category * A, height, width), dtype='float32')
        batch_bbox_targets = np.zeros((n, num_category * 4 * A, height, width), dtype='float32')
        batch_bbox_weights = np.zeros((n, num_category * 4 * A, height, width), dtype='float32')
        # every sample
        for i in range(n):
            gt_box = gt_boxes[i, :4][np.newaxis]  # 1 x 4
            cate_idx = int(gt_boxes[i, -1])
            # label: 1 is positive, 0 is negative, -1 is dont care
            labels = np.empty((len(inds_inside), ), dtype='float32')
            labels.fill(-1)
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps_cython(
                np.ascontiguousarray(anchors, dtype='float'),
                np.ascontiguousarray(gt_box, dtype='float'))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < self.anchor_negative_overlap] = 0
            # fg label: above threshold IOU
            labels[max_overlaps >= self.anchor_positive_overlap] = 1

            # Subsample positives
            num_fg = int(self.anchor_fg_fraction * self.anchor_per_sample)
            fg_inds = np.where(labels == 1)[0]
            if len(fg_inds) > num_fg:
                if self.hard_mining:
                    offset = cate_idx * self.num_anchors * 2 + self.num_anchors
                    # positive score for this category
                    pos_scores = rpn_cls[i, offset: offset + self.num_anchors, :, :]
                    pos_scores = pos_scores.transpose((1, 2, 0)).flatten()
                    pos_scores = pos_scores[inds_inside]
                    pos_scores = pos_scores[fg_inds]
                    order_pos_scores = pos_scores.ravel().argsort()
                    ohem_sampled_fgs = fg_inds[order_pos_scores[:num_fg]]
                    labels[fg_inds] = -1
                    labels[ohem_sampled_fgs] = 1
                else:
                    disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                    labels[disable_inds] = -1
            # Subsample negatives
            num_fg = np.sum(labels == 1)
            num_bg = self.anchor_per_sample - num_fg
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
                if self.hard_mining:
                    offset = cate_idx * self.num_anchors * 2
                    # negative score for this category
                    neg_scores = rpn_cls[i, offset: offset + self.num_anchors, :, :]
                    neg_scores = neg_scores.transpose((1, 2, 0)).flatten()
                    neg_scores = neg_scores[inds_inside]
                    neg_scores = neg_scores[bg_inds]
                    order_neg_scores = neg_scores.ravel().argsort()[::-1]
                    ohem_sampled_bgs = bg_inds[order_neg_scores[:num_bg]]
                    labels[bg_inds] = -1
                    labels[ohem_sampled_bgs] = 0
                else:
                    disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                    labels[disable_inds] = -1

            # bbox target
            bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
            bbox_weights = np.zeros((len(inds_inside), 4), dtype='float32')
            bbox_weights[labels == 1, :] = 1

            # map up to original set of anchors
            labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)
            labels = labels.reshape((height, width, A)).transpose((2, 0, 1))
            bbox_targets = bbox_targets.reshape((height, width, A*4)).transpose((2, 0, 1))
            bbox_weights = bbox_weights.reshape((height, width, A*4)).transpose((2, 0, 1))

            # place in batch
            offset_label = cate_idx * A
            offset_bbox = cate_idx * (A*4)
            batch_labels[i] = -1
            batch_labels[i, offset_label: offset_label + A] = labels
            batch_bbox_targets[i, offset_bbox: offset_bbox + A*4] = bbox_targets
            batch_bbox_weights[i, offset_bbox: offset_bbox + A*4] = bbox_weights

        # attach weight to not used
        batch_labels_weight = np.ones_like(batch_labels, dtype='float32')
        batch_labels_weight[batch_labels == -1] = 0
        batch_labels[batch_labels == -1] = 0
        # move to mxnet ndarray
        batch_labels = nd.array(batch_labels, ctx)
        batch_labels_weight = nd.array(batch_labels_weight, ctx)
        batch_bbox_targets = nd.array(batch_bbox_targets, ctx)
        batch_bbox_weights = nd.array(batch_bbox_weights, ctx)
        return batch_labels, batch_labels_weight, batch_bbox_targets, batch_bbox_weights

    def proposal(self, rpn_cls, rpn_reg, im_info=(368, 368)):
        n, _, height, width = rpn_cls.shape
        num_category = 5
        rpn_cls = rpn_cls.asnumpy()
        rpn_reg = rpn_reg.asnumpy()
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.shape[0]
        anchors = (self.anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        anchors = anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # N, num_category * num_anchor * h * w, 4
        bbox_deltas = rpn_reg.transpose((0, 2, 3, 1)).reshape((n, num_category * total_anchors, 4))
        # N, num_category * num_anchor * h * w
        select = np.hstack([np.arange(i*2*A + A, i*2*A + 2*A) for i in range(num_category)])
        scores = rpn_cls.transpose((0, 2, 3, 1))[:, :, :, select]
        scores = scores.reshape((n, num_category * total_anchors))

        # shape: N, num_category, (proposals, scores)
        dets = []
        for i in range(n):
            res = []
            for j in range(num_category):
                # get proposals
                offset = j * total_anchors
                proposals = bbox_transform_inv(anchors, bbox_deltas[i, offset: offset + total_anchors])
                proposals = clip_boxes(proposals, im_info)
                keep = _filter_boxes(proposals, self.min_size)
                proposals = proposals[keep]
                # get scores
                category_scores = scores[i, offset: offset + total_anchors]
                order = category_scores.ravel().argsort()[::-1]
                order = order[:self.num_det_per_category]
                proposals = proposals[order]
                category_scores = category_scores[order]
                # nms
                keep = cpu_nms(np.hstack([proposals, category_scores.reshape((-1, 1))]), self.nms_th)
                proposals = proposals[keep]
                category_scores = category_scores[keep]
                # result
                res.append((proposals, category_scores))
            dets.append(res)
        return dets


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype='float32')
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype='float32')
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype('float32', copy=False)


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def main():
    anchor_proposal = AnchorProposal(scales=[4, 8], ratios=[1, 0.5, 2], feat_stride=16)
    print(anchor_proposal.anchors)
    ctx = mx.cpu()
    rpn_cls = nd.random.normal(shape=(3, 2*5*anchor_proposal.num_anchors, 23, 23), ctx=ctx)
    gt_box = nd.array([[1, 20, 100, 300, 1], [200, 120, 300, 340, 4], [123, 234, 234, 300, 3]], ctx=ctx)
    batch_labels, batch_labels_weight, batch_bbox_targets, batch_bbox_weights = anchor_proposal.target(rpn_cls, gt_box)
    rpn_reg = nd.zeros(shape=(3, 4*5*anchor_proposal.num_anchors, 23, 23), ctx=ctx)
    dets = anchor_proposal.proposal(rpn_cls, rpn_reg)
    print(dets)


if __name__ == '__main__':
    main()
