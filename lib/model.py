from __future__ import print_function, division

import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn

from lib.config import cfg
from lib.utils import process_cv_img, crop_patch_refine


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
        # freeze_bn(backbone.features)
        # create symbol
        data = mx.sym.var('data')
        out_names = ['_'.join([backbone.name, featname, 'output']) for featname in featnames]
        internals = backbone(data).get_internals()
        outs = [internals[out_name] for out_name in out_names]
        net.backbone = gl.SymbolBlock(outs, data, params=backbone.collect_params())


class ConvBnReLU(gl.HybridBlock):

    def __init__(self, num_channel, kernel_size=3, padding=1):
        super(ConvBnReLU, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            self.body.add(nn.Conv2D(num_channel, kernel_size=kernel_size, padding=padding, use_bias=False))
            self.body.add(nn.BatchNorm())
            self.body.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return x


class ContextBlock(gl.HybridBlock):

    def __init__(self, num_channel):
        super(ContextBlock, self).__init__()
        with self.name_scope():
            self.f3 = ConvBnReLU(num_channel)
            self.f5 = nn.HybridSequential()
            self.f5.add(ConvBnReLU(num_channel), ConvBnReLU(num_channel))
            self.f7 = nn.HybridSequential()
            self.f7.add(ConvBnReLU(num_channel), ConvBnReLU(num_channel), ConvBnReLU(num_channel))

    def hybrid_forward(self, F, x):
        f3 = self.f3(x)
        f5 = self.f5(x)
        f7 = self.f7(x)
        out = F.concat(f3, f5, f7)
        return out


class KpsPafBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel):
        super(KpsPafBlock, self).__init__()
        with self.name_scope():
            self.kps = nn.HybridSequential()
            self.kps.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.kps.add(nn.Conv2D(num_kps, kernel_size=1))
            self.paf = nn.HybridSequential()
            self.paf.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.paf.add(nn.Conv2D(num_limb, kernel_size=1))

    def hybrid_forward(self, F, x):
        ht = self.kps(x)
        paf = self.paf(x)
        return ht, paf


##### model for version 2

class CPMBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel, num_context):
        super(CPMBlock, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            for _ in range(num_context):
                self.body.add(ContextBlock(128))
            self.out = KpsPafBlock(num_kps, num_limb, num_channel)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        ht, paf = self.out(x)
        return x, ht, paf


class PoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_stage, num_channel, num_context=2):
        super(PoseNet, self).__init__(prefix='posenet')
        self.num_stage = num_stage
        self.scale = 3
        with self.name_scope():
            # backbone
            self.backbone = None
            # feature transfer
            self.feat_trans = ConvBnReLU(256)
            # cpm
            self.cpm = nn.HybridSequential()
            for _ in range(num_stage):
                self.cpm.add(CPMBlock(num_kps, num_limb, num_channel, num_context))

    def hybrid_forward(self, F, x):
        x = self.backbone(x)  # pylint: disable=not-callable
        x = self.feat_trans(x)
        feat = x
        outs = []
        for i in range(self.num_stage):
            feat, ht, paf = self.cpm[i](feat)
            outs.append([ht, paf])
            if i != self.num_stage - 1:
                feat = F.concat(feat, x)
                # mask
                mask = F.exp(self.scale * F.max(ht, axis=1, keepdims=True))
                feat = F.broadcast_mul(feat, mask)
        return outs

    def init_backbone(self, creator, featnames, fixed, pretrained=True):
        assert len(featnames) == 1
        install_backbone(self, creator, featnames, fixed, pretrained)

    def predict(self, img, ctx, flip=True):
        data = process_cv_img(img)
        if flip:
            data_flip = data[:, :, ::-1]
            data = np.concatenate([data[np.newaxis], data_flip[np.newaxis]])
        else:
            data = data[np.newaxis]
        batch = mx.nd.array(data, ctx=ctx)
        out = self(batch)
        idx = -1
        heatmap = out[idx][0][0].asnumpy().astype('float64')
        paf = out[idx][1][0].asnumpy().astype('float64')
        if flip:
            heatmap_flip = out[idx][0][1].asnumpy().astype('float64')
            paf_flip = out[idx][1][1].asnumpy().astype('float64')
            heatmap, paf = flip_prediction(heatmap, heatmap_flip, paf, paf_flip)
        return heatmap, paf


##### model for version 3

class UpSamplingBlock(gl.HybridBlock):

    def __init__(self, num_channel, scale):
        super(UpSamplingBlock, self).__init__()
        with self.name_scope():
            self.body = nn.Conv2DTranspose(num_channel, kernel_size=2*scale, strides=scale, padding=scale//2,
                                           use_bias=False, groups=num_channel, weight_initializer=mx.init.Bilinear())
            self.body.collect_params().setattr('lr_mult', 0)

    def hybrid_forward(self, F, x):
        return self.body(x)


class CPNGlobalBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel):
        super(CPNGlobalBlock, self).__init__()
        with self.name_scope():
            self.P4 = nn.Conv2D(128, kernel_size=1, activation='relu')   # 1/4
            self.P8 = nn.Conv2D(128, kernel_size=1, activation='relu')   # 1/8
            self.P16 = nn.Conv2D(128, kernel_size=1, activation='relu')  # 1/16
            self.upx2 = UpSamplingBlock(128, 2)
            self.T8 = nn.Conv2D(128, kernel_size=1, activation='relu')  # 1/16 -> 1/8
            self.T4 = nn.Conv2D(128, kernel_size=1, activation='relu')  # 1/8 -> 1/4
            self.Pre4 = nn.HybridSequential()
            self.Pre4.add(ContextBlock(64), ContextBlock(64), KpsPafBlock(num_kps, num_limb, num_channel))
            self.Pre8 = nn.HybridSequential()
            self.Pre8.add(ContextBlock(128), ContextBlock(128), KpsPafBlock(num_kps, num_limb, num_channel))
            self.Pre16 = nn.HybridSequential()
            self.Pre16.add(ContextBlock(128), ContextBlock(128), KpsPafBlock(num_kps, num_limb, num_channel))

    def hybrid_forward(self, F, f4, f8, f16):
        f4 = self.P4(f4)
        f8 = self.P8(f8)
        f16 = self.P16(f16)
        # 1/16
        ht16, paf16 = self.Pre16(f16)
        # 1/8
        u8 = F.Crop(self.upx2(f16), f8)
        f8 = f8 + self.T8(u8)
        ht8, paf8 = self.Pre8(f8)
        # 1/4
        u4 = F.Crop(self.upx2(f8), f4)
        f4 = f4 + self.T4(u4)
        ht4, paf4 = self.Pre4(f4)
        return f4, f8, f16, ht4, ht8, ht16, paf4, paf8, paf16


class CPNRefineBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel):
        super(CPNRefineBlock, self).__init__()
        with self.name_scope():
            self.upx2 = UpSamplingBlock(num_channel, 2)
            self.upx4 = UpSamplingBlock(num_channel, 4)
            self.feat_trans = nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu')
            self.pred = nn.HybridSequential()
            self.pred.add(ContextBlock(num_channel), ContextBlock(num_channel))
            self.pred.add(KpsPafBlock(num_kps, num_limb, num_channel))

    def hybrid_forward(self, F, f4, f8, f16):
        u8 = F.Crop(self.upx2(f8), f4)
        u16 = F.Crop(self.upx4(f16), f4)
        x = F.concat(f4, u8, u16)
        x = self.feat_trans(x)
        ht, paf = self.pred(x)
        return ht, paf


class CascadePoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel):
        super(CascadePoseNet, self).__init__(prefix='cpn')
        with self.name_scope():
            self.backbone = None
            self.f4 = ConvBnReLU(256)
            self.f8 = ConvBnReLU(256)
            self.f16 = ConvBnReLU(256)
            self.upx2 = UpSamplingBlock(256, 2)
            self.context16 = nn.HybridSequential()
            self.context16.add(ContextBlock(128), ContextBlock(128))
            self.context16tofeat = ConvBnReLU(256)
            self.p16_1 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.p16_2 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.context8 = nn.HybridSequential()
            self.context8.add(ContextBlock(128), ContextBlock(128))
            self.context8tofeat = ConvBnReLU(256)
            self.p8_1 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.p8_2 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.context4 = nn.HybridSequential()
            self.context4.add(ContextBlock(64), ContextBlock(64))
            self.p4_1 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.p4_2 = KpsPafBlock(num_kps, num_limb, num_channel)


    def hybrid_forward(self, F, x):
        f4, f8, f16 = self.backbone(x)  # pylint: disable=not-callable
        # 16
        f16 = self.f16(f16)
        g_ht16, g_paf16 = self.p16_1(f16)
        f16 = self.context16(f16)
        r_ht16, r_paf16 = self.p16_2(f16)
        # 8
        f16 = self.context16tofeat(f16)
        f8 = self.f8(f8) + F.Crop(self.upx2(f16), f8)
        g_ht8, g_paf8 = self.p8_1(f8)
        f8 = self.context8(f8)
        r_ht8, r_paf8 = self.p8_2(f8)
        # 4
        f8 = self.context8tofeat(f8)
        f4 = self.f4(f4) + F.Crop(self.upx2(f8), f4)
        g_ht4, g_paf4 = self.p4_1(f4)
        f4 = self.context4(f4)
        r_ht4, r_paf4 = self.p4_2(f4)
        return g_ht4, g_paf4, r_ht4, r_paf4, g_ht8, g_paf8, r_ht8, r_paf8, g_ht16, g_paf16, r_ht16, r_paf16

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
        g_ht4, g_paf4, r_ht4, r_paf4, g_ht8, g_paf8, r_ht8, r_paf8, g_ht16, g_paf16, r_ht16, r_paf16 = self(batch)
        heatmap = g_ht4[0].asnumpy().astype('float64')
        paf = g_paf4[0].asnumpy().astype('float64')
        if flip:
            heatmap_flip = g_ht4[1].asnumpy().astype('float64')
            paf_flip = g_paf4[1].asnumpy().astype('float64')
            heatmap, paf = flip_prediction(heatmap, heatmap_flip, paf, paf_flip)
        return heatmap, paf


def multi_scale_predict(net, ctx, img, multi_scale=False):
    if multi_scale:
        scales = [440, 368, 224]
    else:
        scales = [368]
    h, w = img.shape[:2]
    # init
    heatmap = 0
    paf = 0
    for scale in scales:
        factor = scale / max(h, w)
        img_ = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        heatmap_, paf_ = net.predict(img_, ctx)
        heatmap_ = cv2.resize(heatmap_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
        paf_ = cv2.resize(paf_.transpose((1, 2, 0)), (h, w), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
        heatmap = heatmap + heatmap_
        paf = paf + paf_
    heatmap /= len(scales)
    paf /= len(scales)
    return heatmap, paf


##### Utils

def flip_prediction(ht, ht_flip, paf, paf_flip):
    # heatmap
    ht_flip = ht_flip[:, :, ::-1]
    for i, j in cfg.LANDMARK_SWAP:
        tmp = ht_flip[i].copy()
        ht_flip[i] = ht_flip[j]
        ht_flip[j] = tmp
    ht = (ht + ht_flip) / 2
    # paf
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
                tmp = paf_flip[i].copy()
                paf_flip[i] = paf_flip[j]
                paf_flip[j] = tmp
            elif p1 == p4 and p2 == p3:
                assert i == j
    paf = (paf + paf_flip) / 2
    return ht, paf


def parse_from_name_v2(name):
    # name = /path/to/V2.default-vgg16-S5-C64-C2-BS16-adam-0100.params
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    ps = name.split('-')
    prefix = ps[0]
    backbone = ps[1]
    stages = int(ps[2][1:])
    channels = int(ps[3][1:])
    contexts = int(ps[4][1:])
    batch_size = int(ps[5][2:])
    optim = ps[6]
    epoch = int(ps[7])
    return prefix, backbone, stages, channels, contexts, batch_size, optim, epoch


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
        prefix, backbone, num_stage, num_channel, num_context, batch_size, optim, epoch = parse_from_name_v2(model)
        net = PoseNet(num_kps=num_kps, num_limb=num_limb, num_stage=num_stage, num_channel=num_channel, num_context=num_context)
        creator, featnames, fixed = cfg.BACKBONE_v2[backbone]
    elif version == 3:
        prefix, backbone, num_channel, batch_size, optim, epoch = parse_from_name_v3(model)
        net = CascadePoseNet(num_kps=num_kps, num_limb=num_limb, num_channel=num_channel)
        creator, featnames, fixed = cfg.BACKBONE_v3[backbone]
    else:
        raise RuntimeError('no such version %d'%version)
    net.init_backbone(creator, featnames, fixed, pretrained=False)
    net.load_params(model, mx.cpu(0))
    return net


##### detection model

class FPNBlock(gl.HybridBlock):

    def __init__(self, num_channel):
        super(FPNBlock, self).__init__()
        with self.name_scope():
            self.P8 = nn.Conv2D(num_channel, kernel_size=1, padding=1, activation='relu')
            self.P16 = nn.Conv2D(num_channel, kernel_size=1, padding=1, activation='relu')
            self.up = UpSamplingBlock(num_channel, 2)

    def hybrid_forward(self, F, f8, f16):
        f8 = self.P8(f8)
        f16 = self.P16(f16)
        f8 = f8 + F.Crop(self.up(f16), f8)
        return f8, f16


class RPNBlock(gl.HybridBlock):

    def __init__(self, num_anchors, num_category):
        super(RPNBlock, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            self.body.add(nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
            # rpn_cls: N, C x A x 2, H, W
            # rpn_reg: N, C X A x 4, H, W
            self.rpn_cls = nn.Conv2D(2*num_anchors*num_category, kernel_size=1)
            self.rpn_reg = nn.Conv2D(4*num_anchors*num_category, kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        anchor_cls = self.rpn_cls(x)
        anchor_reg = self.rpn_reg(x)
        return anchor_cls, anchor_reg


class DetNet(gl.HybridBlock):

    def __init__(self, anchor_proposals):
        super(DetNet, self).__init__(prefix='detnet')
        assert len(anchor_proposals) == 2
        num_category = 3
        self.anchor_proposals = anchor_proposals
        with self.name_scope():
            self.backbone = None
            self.fpn = FPNBlock(num_channel=128)
            self.rpn1 = RPNBlock(anchor_proposals[0].num_anchors, num_category)
            self.rpn2 = RPNBlock(anchor_proposals[1].num_anchors, num_category)

    def hybrid_forward(self, F, x):
        f8, f16 = self.backbone(x)  # pylint: disable=not-callable
        f8, f16 = self.fpn(f8, f16)
        anchor_cls1, anchor_reg1 = self.rpn1(f8)
        anchor_cls2, anchor_reg2 = self.rpn2(f16)
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
        dets2 = self.anchor_proposals[1].proposal(anchor_cls2, anchor_reg2, (h, w), nms)[0]
        dets = [np.vstack([x, y]) for x, y in zip(dets1, dets2)]
        if nms:
            dets = [self.anchor_proposals[0].nms(det) for det in dets]
        return dets


def multi_scale_detection(net, ctx, img, category, multi_scale=False):
    if multi_scale:
        scales = [440, 368, 224]
    else:
        scales = [368]
    h, w = img.shape[:2]
    dets = []
    cate_idx = cfg.DET_CATE[category]
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
