from __future__ import print_function, division

import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn

from lib.config import cfg
from lib.utils import process_cv_img


def freeze_bn(block):
    if isinstance(block, nn.BatchNorm):
        print('freeze batchnorm operator', block.name)
        block._kwargs['use_global_stats'] = True
        block.gamma.grad_req = 'null'
        block.beta.grad_req = 'null'
    else:
        for child in block._children.values():
            freeze_bn(child)


def install_backbone(net, creator, featnames, fixed, pretrained):
    with net.name_scope():
        backbone = creator(pretrained=pretrained)
        name = backbone.name
        # hacking parameters
        params = backbone.collect_params()
        for key, item in params.items():
            should_fix = False
            for pattern in fixed:
                if name + '_' + pattern + '_' in key:
                    should_fix = True
            if should_fix:
                print('fix parameter', key)
                item.grad_req = 'null'
        # special for batchnorm
        freeze_bn(backbone.features)
        # create symbol
        data = mx.sym.var('data')
        out_names = ['_'.join([backbone.name, featname, 'output']) for featname in featnames]
        internals = backbone(data).get_internals()
        outs = [internals[out_name] for out_name in out_names]
        net.backbone = gl.SymbolBlock(outs, data, params=backbone.collect_params())


##### model for version 2

class CPMBlock(gl.HybridBlock):

    def __init__(self, num_output, channels, ks=[3, 3, 3, 1, 1]):
        super(CPMBlock, self).__init__()
        with self.name_scope():
            self.net = nn.HybridSequential()
            for k in ks[:-1]:
                self.net.add(nn.Conv2D(channels, k, 1, k // 2, activation='relu'))
            self.net.add(nn.Conv2D(num_output, ks[-1], 1, ks[-1] // 2))
            for conv in self.net:
                conv.weight.lr_mult = 4
                conv.bias.lr_mult = 8
                #conv.weight.wd_mult = 4

    def hybrid_forward(self, F, x):
        return self.net(x)


class PoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, stages, channels):
        super(PoseNet, self).__init__()
        with self.name_scope():
            # backbone
            self.backbone = None
            # feature transfer
            self.feature_trans = nn.HybridSequential()
            self.feature_trans.add(nn.Conv2D(2*channels, 3, 1, 1, activation='relu'),
                                   nn.Conv2D(channels, 3, 1, 1, activation='relu'))
            # cpm
            self.stages = stages
            self.kps_cpm = nn.HybridSequential()
            self.limb_cpm = nn.HybridSequential()
            ks1 = [3, 3, 3, 1, 1]
            ks2 = [7, 7, 7, 7, 7, 1, 1]
            self.kps_cpm.add(CPMBlock(num_kps + 1, channels, ks1))
            self.limb_cpm.add(CPMBlock(2*num_limb, channels, ks1))
            for _ in range(1, stages):
                self.kps_cpm.add(CPMBlock(num_kps + 1, channels, ks2))
                self.limb_cpm.add(CPMBlock(2*num_limb, channels, ks2))

    def hybrid_forward(self, F, x):
        feat = self.backbone(x)  # pylint: disable=not-callable
        feat = self.feature_trans(feat)
        out = feat
        outs = []
        for i in range(self.stages):
            out1 = self.kps_cpm[i](out)
            out2 = self.limb_cpm[i](out)
            outs.append([out1, out2])
            out1 = F.stop_gradient(out1)
            out2 = F.stop_gradient(out2)
            out = F.concat(feat, out1, out2)
        return outs

    def init_backbone(self, creator, featname, fixed, pretrained=True):
        install_backbone(self, creator, [featname], fixed, pretrained)

    def predict(self, img, ctx, flip=True):
        data = process_cv_img(img)
        if flip:
            data_flip = data[:, :, ::-1]
            data = np.concatenate([data[np.newaxis], data_flip[np.newaxis]])
        else:
            data = data[np.newaxis]
        batch = mx.nd.array(data, ctx=ctx)
        out = self(batch)
        heatmap = out[-1][0][0].asnumpy().astype('float64')
        paf = out[-1][1][0].asnumpy().astype('float64')
        if flip:
            # heatmap
            heatmap_flip = out[-1][0][1].asnumpy().astype('float64')
            heatmap_flip = heatmap_flip[:, :, ::-1]
            for i, j in cfg.LANDMARK_SWAP:
                tmp = heatmap_flip[i].copy()
                heatmap_flip[i] = heatmap_flip[j]
                heatmap_flip[j] = tmp
            heatmap = (heatmap + heatmap_flip) / 2
            # paf
            paf_flip = out[-1][1][1].asnumpy().astype('float64')
            paf_flip = paf_flip[:, :, ::-1]

            def get_flip_idx(p):
                for px, py in cfg.LANDMARK_SWAP:
                    if px == p:
                        return py
                    if py == p:
                        return px
                return p

            num = len(cfg.PAF_LANDMARK_PAIR)
            for i in range(num):
                p1, p2 = cfg.PAF_LANDMARK_PAIR[i]
                p1, p2 = get_flip_idx(p1), get_flip_idx(p2)
                for j in range(i, num):
                    p3, p4 = cfg.PAF_LANDMARK_PAIR[j]
                    if p1 == p3 and p2 == p4:
                        tmp = paf_flip[2*i: 2*i+2].copy()
                        paf_flip[2*i: 2*i+2] = paf_flip[2*j: 2*j+2]
                        paf_flip[2*j: 2*j+2] = tmp
                        paf_flip[2*i] *= -1  # flip x
                        paf_flip[2*j] *= -1
                    elif p1 == p4 and p2 == p3:
                        assert i == j
                        paf_flip[2*i+1] *= -1  # flip y
            paf = (paf + paf_flip) / 2

        return heatmap, paf


##### model for version 3

class UpSampling(gl.HybridBlock):

    def __init__(self, num_channel, scale):
        super(UpSampling, self).__init__()
        with self.name_scope():
            self.body = nn.Conv2DTranspose(num_channel, kernel_size=2*scale, strides=scale, padding=scale//2,
                                           use_bias=False, groups=num_channel, weight_initializer=mx.init.Bilinear())
            self.body.collect_params().setattr('lr_mult', 0)

    def hybrid_forward(self, F, x):
        return self.body(x)


class CPNGlobalBlock(gl.HybridBlock):

    def __init__(self, num_output, num_channel):
        super(CPNGlobalBlock, self).__init__()
        self.num_output = num_output
        self.num_channel = num_channel
        with self.name_scope():
            self.P4 = nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu')   # 1/4
            self.P8 = nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu')   # 1/8
            self.P16 = nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu')  # 1/16
            self.UpSampling = UpSampling(num_channel, 2)
            self.T8 = nn.Conv2D(num_channel, 1, 1, 0)  # 1/16 -> 1/8
            self.T4 = nn.Conv2D(num_channel, 1, 1, 0)  # 1/8 -> 1/4
            self.Pre4 = nn.HybridSequential()
            self.Pre4.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.Pre4.add(nn.Conv2D(num_output, kernel_size=3, padding=1))
            self.Pre8 = nn.HybridSequential()
            self.Pre8.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.Pre8.add(nn.Conv2D(num_output, kernel_size=3, padding=1))
            self.Pre16 = nn.HybridSequential()
            self.Pre16.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.Pre16.add(nn.Conv2D(num_output, kernel_size=3, padding=1))

    def hybrid_forward(self, F, x4, x8, x16):
        F4 = self.P4(x4)
        F8 = self.P8(x8)
        F16 = self.P16(x16)
        # 1/16
        R16 = self.Pre16(F16)
        # 1/8
        U8 = self.UpSampling(F16)
        U8 = F.Crop(U8, F8)
        F8 = F8 + self.T8(U8)
        R8 = self.Pre8(F8)
        # 1/4
        U4 = self.UpSampling(F8)
        U4 = F.Crop(U4, F4)
        F4 = F4 + self.T4(U4)
        R4 = self.Pre4(F4)
        return F4, F8, F16, R4, R8, R16


class CPNRefineBlock(gl.HybridBlock):

    def __init__(self, num_output, num_channel):
        super(CPNRefineBlock, self).__init__()
        with self.name_scope():
            self.feat_trans = nn.HybridSequential()
            ks = [7, 7, 7, 7, 7]
            for k in ks:
                self.feat_trans.add(nn.Conv2D(num_channel, k, 1, k // 2, activation='relu'))
            self.heat_pred = nn.HybridSequential()
            self.heat_pred.add(nn.Conv2D(num_channel, 3, 1, 1, activation='relu'),
                               nn.Conv2D(num_output, 3, 1, 1))

    def hybrid_forward(self, F, x):
        feat = self.feat_trans(x)
        out = self.heat_pred(feat)
        return out


class CascadePoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_channel):
        super(CascadePoseNet, self).__init__()
        self.num_kps = num_kps
        self.num_channel = num_channel
        with self.name_scope():
            self.backbone = None
            self.global_net = CPNGlobalBlock(self.num_kps + 1, self.num_channel)
            self.refine_net = CPNRefineBlock(self.num_kps + 1, self.num_channel)

    def hybrid_forward(self, F, x):
        feats = self.backbone(x)  # pylint: disable=not-callable
        F4, F8, F16, R4, R8, R16 = self.global_net(*feats)
        refine_pred = self.refine_net(F.concat(F4, R4))
        return [R4, R8, R16], refine_pred

    def init_backbone(self, creator, featnames, fixed, pretrained=True):
        assert len(featnames) == 3
        install_backbone(self, creator, featnames, fixed, pretrained)

    def predict(self, img, ctx, flip=True):
        data = process_cv_img(img)
        if flip:
            data_flip = data[:, :, ::-1]
            data = np.concatenate([data[np.newaxis], data_flip[np.newaxis]])
        else:
            data = data[np.newaxis]
        batch = mx.nd.array(data, ctx=ctx)
        _, refine_pred = self(batch)
        heatmap = refine_pred[0].asnumpy().astype('float64')
        if flip:
            heatmap_flip = refine_pred[1].asnumpy().astype('float64')
            heatmap_flip = heatmap_flip[:, :, ::-1]
            for i, j in cfg.LANDMARK_SWAP:
                tmp = heatmap_flip[i].copy()
                heatmap_flip[i] = heatmap_flip[j]
                heatmap_flip[j] = tmp
            heatmap = (heatmap + heatmap_flip) / 2
        return heatmap


##### model for version 4

class MaskHeatHead(gl.HybridBlock):

    def __init__(self, num_output, num_channel):
        super(MaskHeatHead, self).__init__()
        with self.name_scope():
            self.feat_trans = nn.HybridSequential()
            ks = [7, 7, 7, 7, 7]
            for k in ks:
                self.feat_trans.add(nn.Conv2D(num_channel, k, 1, k // 2, activation='relu'))
            self.mask_pred = nn.HybridSequential()
            self.mask_pred.add(nn.Conv2D(num_channel, 3, 1, 1, activation='relu'),
                               nn.Conv2D(num_channel, 3, 1, 1, activation='relu'),
                               nn.Conv2D(num_channel, 3, 1, 1, activation='relu'),
                               nn.Conv2D(5, 3, 1, 1))
            self.heat_pred = nn.HybridSequential()
            self.heat_pred.add(nn.Conv2D(num_channel, 3, 1, 1, activation='relu'),
                               nn.Conv2D(num_output, 3, 1, 1))

    def hybrid_forward(self, F, x):
        feat = self.feat_trans(x)
        mask = self.mask_pred(feat)
        mask_to_feat = F.sigmoid(mask)
        mask_to_feat = F.max(mask_to_feat, axis=1, keepdims=True)
        mask_to_feat = F.exp(mask_to_feat)
        mask_to_feat = F.stop_gradient(mask_to_feat)
        feat = F.broadcast_mul(feat, mask_to_feat)
        heatmap = self.heat_pred(feat)
        return feat, mask, mask_to_feat, heatmap


class RefineNet(gl.HybridBlock):

    def __init__(self, num_output, num_channel):
        super(RefineNet, self).__init__()
        with self.name_scope():
            self.feat_trans = nn.HybridSequential()
            ks = [7, 7, 7, 7, 7]
            for k in ks:
                self.feat_trans.add(nn.Conv2D(num_channel, k, 1, k // 2, activation='relu'))
            self.heat_pred = nn.HybridSequential()
            self.heat_pred.add(nn.Conv2D(num_channel, 3, 1, 1, activation='relu'),
                               nn.Conv2D(num_output, 3, 1, 1))

    def hybrid_forward(self, F, x, mask_to_feat):
        feat = self.feat_trans(x)
        feat = F.broadcast_mul(feat, mask_to_feat)
        heatmap = self.heat_pred(feat)
        return heatmap


class MaskPoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_channel):
        super(MaskPoseNet, self).__init__()
        with self.name_scope():
            self.backbone = None
            self.head = MaskHeatHead(num_kps + 1, num_channel)
            self.refine = RefineNet(num_kps + 1, num_channel)

    def hybrid_forward(self, F, x):
        feat = self.backbone(x)  # pylint: disable=not-callable
        feat_with_mask, mask, mask_to_feat, heatmap = self.head(feat)
        feat = F.concat(feat, feat_with_mask, heatmap)
        heatmap_refine = self.refine(feat, mask_to_feat)
        return mask, heatmap, heatmap_refine

    def init_backbone(self, creator, featname, fixed, pretrained=True):
        install_backbone(self, creator, [featname], fixed, pretrained)

    def predict(self, img, ctx, flip=True):
        data = process_cv_img(img)
        data = data[np.newaxis]
        batch = mx.nd.array(data, ctx=ctx)
        mask, _, heatmap = self(batch)
        mask = nd.sigmoid(mask)
        heatmap = heatmap[0].asnumpy().astype('float64')
        mask = mask[0].asnumpy().astype('float64')
        return mask, heatmap


##### model for version 5

class PoseNetNoPaf(gl.HybridBlock):

    def __init__(self, num_kps, stages, channels):
        super(PoseNetNoPaf, self).__init__()
        with self.name_scope():
            # backbone
            self.backbone = None
            # feature transfer
            self.feature_trans = nn.HybridSequential()
            self.feature_trans.add(nn.Conv2D(2*channels, 3, 1, 1, activation='relu'),
                                   nn.Conv2D(channels, 3, 1, 1, activation='relu'))
            # cpm
            self.stages = stages
            self.kps_cpm = nn.HybridSequential()
            ks1 = [3, 3, 3, 1, 1]
            ks2 = [7, 7, 7, 7, 7, 1, 1]
            self.kps_cpm.add(CPMBlock(num_kps + 1, channels, ks1))
            for _ in range(1, stages):
                self.kps_cpm.add(CPMBlock(num_kps + 1, channels, ks2))

    def hybrid_forward(self, F, x):
        feat = self.backbone(x)  # pylint: disable=not-callable
        feat = self.feature_trans(feat)
        out = feat
        outs = []
        for i in range(self.stages):
            out = self.kps_cpm[i](out)
            outs.append(out)
            out = F.concat(feat, out)
        return outs

    def init_backbone(self, creator, featname, fixed, pretrained=True):
        install_backbone(self, creator, [featname], fixed, pretrained)

    def predict(self, img, ctx, flip=True):
        data = process_cv_img(img)
        if flip:
            data_flip = data[:, :, ::-1]
            data = np.concatenate([data[np.newaxis], data_flip[np.newaxis]])
        else:
            data = data[np.newaxis]
        batch = mx.nd.array(data, ctx=ctx)
        out = self(batch)
        heatmap = out[-1][0].asnumpy().astype('float64')
        if flip:
            # heatmap
            heatmap_flip = out[-1][1].asnumpy().astype('float64')
            heatmap_flip = heatmap_flip[:, :, ::-1]
            for i, j in cfg.LANDMARK_SWAP:
                tmp = heatmap_flip[i].copy()
                heatmap_flip[i] = heatmap_flip[j]
                heatmap_flip[j] = tmp
            heatmap = (heatmap + heatmap_flip) / 2

        return heatmap


##### detection model

class GConv(gl.HybridBlock):

    def __init__(self, channels, groups):
        super(GConv, self).__init__()
        self.groups = groups
        with self.name_scope():
            self.body = nn.HybridSequential()
            for _ in range(groups):
                self.body.add(nn.Conv2D(channels, kernel_size=3, padding=1, activation='relu'))

    def hybrid_forward(self, F, x):
        feat = []
        for i in range(self.groups):
            feat.append(self.body[i](x))
        feat = F.concat(*feat)
        return feat


class FPN(gl.HybridBlock):

    def __init__(self, num_channel):
        super(FPN, self).__init__()
        with self.name_scope():
            self.P8 = nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu')
            self.P16 = nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu')
            self.up = UpSampling(num_channel, 2)

    def hybrid_forward(self, F, f8, f16):
        f8 = self.P8(f8)
        f16 = self.P16(f16)
        f8 = f8 + F.Crop(self.up(f16), f8)
        return f8, f16

class RPN(gl.HybridBlock):

    def __init__(self, num_anchors, num_category):
        super(RPN, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            self.body.add(nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
            self.body.add(GConv(64, num_category))
            # rpn_cls: N, C x A x 2, H, W
            # rpn_reg: N, C X A x 4, H, W
            self.rpn_cls = nn.Conv2D(2*num_anchors*num_category, kernel_size=1, groups=num_category)
            self.rpn_reg = nn.Conv2D(4*num_anchors*num_category, kernel_size=1, groups=num_category)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        anchor_cls = self.rpn_cls(x)
        anchor_reg = self.rpn_reg(x)
        return anchor_cls, anchor_reg


class DetNet(gl.HybridBlock):

    def __init__(self, anchor_proposals):
        super(DetNet, self).__init__()
        assert len(anchor_proposals) == 2
        num_category = 5
        self.anchor_proposals = anchor_proposals
        with self.name_scope():
            self.backbone = None
            self.rpn1 = RPN(anchor_proposals[0].num_anchors, num_category)
            self.rpn2 = RPN(anchor_proposals[1].num_anchors, num_category)

    def hybrid_forward(self, F, x):
        feats = self.backbone(x)  # pylint: disable=not-callable
        anchor_cls1, anchor_reg1 = self.rpn1(feats[0])
        anchor_cls2, anchor_reg2 = self.rpn2(feats[1])
        return anchor_cls1, anchor_reg1, anchor_cls2, anchor_reg2

    def init_backbone(self, creator, featnames, fixed, pretrained=True):
        assert len(featnames) == 2
        install_backbone(self, creator, featnames, fixed, pretrained)

    def predict(self, img, ctx, nms=True):
        h, w = img.shape[:2]
        data = process_cv_img(img)
        data = data[np.newaxis]
        batch = mx.nd.array(data, ctx=ctx)
        anchor_cls1, anchor_reg1, anchor_cls2, anchor_reg2 = self(batch)
        dets1 = self.anchor_proposals[0].proposal(anchor_cls1, anchor_reg1, (h, w), nms)[0]
        dets2 = self.anchor_proposals[0].proposal(anchor_cls2, anchor_reg2, (h, w), nms)[0]
        dets = [np.vstack([x, y]) for x, y in zip(dets1, dets2)]
        if nms:
            dets = [self.anchor_proposals[0].nms(det) for det in dets]
        return dets


##### Utils

def parse_from_name_v2(name):
    # name = /path/to/V2.default-vgg16-S5-C64-BS16-adam-0100.params
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    ps = name.split('-')
    prefix = ps[0]
    backbone = ps[1]
    stages = int(ps[2][1:])
    channels = int(ps[3][1:])
    batch_size = int(ps[4][2:])
    optim = ps[5]
    epoch = int(ps[6])
    return prefix, backbone, stages, channels, batch_size, optim, epoch


def parse_from_name_v3(name):
    # name = /path/to/V3.default-resnet50-C64-BS16-adam-0100.params
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    ps = name.split('-')
    prefix = ps[0]
    backbone = ps[1]
    channels = int(ps[2][1:])
    batch_size = int(ps[3][2:])
    optim = ps[4]
    epoch = int(ps[5])
    return prefix, backbone, channels, batch_size, optim, epoch


def load_model(model, version=2):
    num_kps = cfg.NUM_LANDMARK
    num_limb = len(cfg.PAF_LANDMARK_PAIR)
    if version == 2:
        prefix, backbone, cpm_stages, cpm_channels, batch_size, optim, epoch = parse_from_name_v2(model)
        net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
        creator, featname, fixed = cfg.BACKBONE_v2[backbone]
    elif version == 3:
        prefix, backbone, num_channel, batch_size, optim, epoch = parse_from_name_v3(model)
        net = CascadePoseNet(num_kps=num_kps, num_channel=num_channel)
        creator, featname, fixed = cfg.BACKBONE_v3[backbone]
    elif version == 4:
        prefix, backbone, num_channel, batch_size, optim, epoch = parse_from_name_v3(model)
        net = MaskPoseNet(num_kps=num_kps, num_channel=num_channel)
        creator, featname, fixed = cfg.BACKBONE_v4[backbone]
    elif version == 5:
        prefix, backbone, cpm_stages, cpm_channels, batch_size, optim, epoch = parse_from_name_v2(model)
        net = PoseNetNoPaf(num_kps=num_kps, stages=cpm_stages, channels=cpm_channels)
        creator, featname, fixed = cfg.BACKBONE_v2[backbone]
    else:
        raise RuntimeError('no such version %d'%version)
    net.init_backbone(creator, featname, fixed, pretrained=False)
    net.load_params(model, mx.cpu(0))
    return net


def multi_scale_detection(net, ctx, img, category, multi_scale=False):
    if multi_scale:
        scales = [440, 368, 224]
    else:
        scales = [368,]
    h, w = img.shape[:2]
    dets = []
    cate_idx = cfg.CATEGORY.index(category)
    for scale in scales:
        factor = scale / max(h, w)
        img_ = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        dets_ = net.predict(img_, ctx, nms=False)
        dets_ = dets_[cate_idx]
        dets_[:, :4] /= factor
        dets.append(dets_)
    proposals = np.vstack(dets)
    proposals = net.anchor_proposals[0].nms(proposals)
    return proposals


def multi_scale_predict(net, ctx, version, img, category, multi_scale=False):
    if multi_scale:
        scales = [440, 368, 224]
    else:
        scales = [368,]
    h, w = img.shape[:2]
    # init
    heatmap = 0
    paf = 0
    mask = 0
    for scale in scales:
        factor = scale / max(h, w)
        img_ = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        if version == 2:
            heatmap_, paf_ = net.predict(img_, ctx)
            heatmap_ = cv2.resize(heatmap_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            paf_ = cv2.resize(paf_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            heatmap = heatmap + heatmap_
            paf = paf + paf_
        elif version == 3:
            heatmap_ = net.predict(img, ctx)
            heatmap_ = cv2.resize(heatmap_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            heatmap = heatmap + heatmap_
        elif version == 4:
            mask_, heatmap_ = net.predict(img, ctx)
            mask_ = cv2.resize(mask_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            heatmap_ = cv2.resize(heatmap_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            mask = mask + mask_
            heatmap = heatmap + heatmap_
        elif version == 5:
            heatmap_ = net.predict(img_, ctx)
            heatmap_ = cv2.resize(heatmap_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            heatmap = heatmap + heatmap_
        else:
            raise RuntimeError('no such version %d'%version)
    if version == 2:
        heatmap /= len(scales)
        paf /= len(scales)
        return heatmap, paf
    elif version == 3:
        heatmap /= len(scales)
        return heatmap
    elif version == 4:
        mask /= len(scales)
        heatmap /= len(scales)
        return mask, heatmap
    elif version == 5:
        heatmap /= len(scales)
        return heatmap
    else:
        raise RuntimeError('no such version %d'%version)
