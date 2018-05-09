from __future__ import print_function, division

import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn
from mxnet.gluon.model_zoo.vision.resnet import BottleneckV1

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
        # freeze_bn(backbone.features)
        # create symbol
        data = mx.sym.var('data')
        out_names = ['_'.join([backbone.name, featname, 'output']) for featname in featnames]
        internals = backbone(data).get_internals()
        outs = [internals[out_name] for out_name in out_names]
        net.backbone = gl.SymbolBlock(outs, data, params=backbone.collect_params())


class ConvBnReLU(gl.HybridBlock):

    def __init__(self, num_channel):
        super(ConvBnReLU, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            self.body.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, use_bias=False))
            self.body.add(nn.BatchNorm())
            self.body.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return x


class ContextBlock(gl.HybridBlock):

    def __init__(self, num_channel):
        super(ContextBlock, self).__init__()
        with self.name_scope():
            self.conv1 = ConvBnReLU(num_channel)
            self.conv2 = ConvBnReLU(num_channel // 2)
            self.conv3 = ConvBnReLU(num_channel // 2)
            self.conv4_1 = ConvBnReLU(num_channel // 2)
            self.conv4_2 = ConvBnReLU(num_channel // 2)

    def hybrid_forward(self, F, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4_2(self.conv4_1(x2))
        out = F.concat(x1, x3, x4)
        return out


class KpsPafBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel):
        super(KpsPafBlock, self).__init__()
        with self.name_scope():
            self.context = ContextBlock(num_channel)
            self.kps = nn.HybridSequential()
            self.kps.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.kps.add(nn.Conv2D(num_kps, kernel_size=1))
            self.paf = nn.HybridSequential()
            self.paf.add(nn.Conv2D(num_channel, kernel_size=3, padding=1, activation='relu'))
            self.paf.add(nn.Conv2D(num_limb, kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.context(x)
        ht = self.kps(x)
        paf = self.paf(x)
        return ht, paf


class UpSamplingBlock(gl.HybridBlock):

    def __init__(self, num_channel, scale):
        super(UpSamplingBlock, self).__init__()
        with self.name_scope():
            self.body = nn.Conv2DTranspose(num_channel, kernel_size=2*scale, strides=scale, padding=scale//2,
                                           use_bias=False, groups=num_channel, weight_initializer=mx.init.Bilinear())
            self.body.collect_params().setattr('lr_mult', 0)

    def hybrid_forward(self, F, x):
        return self.body(x)


##### model for version 2

class CPMBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel, num_bottleneck):
        super(CPMBlock, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            self.body.add(ConvBnReLU(256))  # reduce channel
            for _ in range(num_bottleneck):
                self.body.add(BottleneckV1(channels=256, stride=1))
            self.out = KpsPafBlock(num_kps, num_limb, num_channel)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        ht, paf = self.out(x)
        return x, ht, paf


class PoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_stage, num_channel):
        super(PoseNet, self).__init__(prefix='posenet')
        self.num_stage = num_stage
        with self.name_scope():
            # backbone
            self.backbone = None
            # feature transfer
            self.feat_trans = ConvBnReLU(num_channel)
            # cpm
            self.cpm = nn.HybridSequential()
            num_bottleneck = [3, 3, 3, 3, 3]
            for i in range(num_stage):
                self.cpm.add(CPMBlock(num_kps, num_limb, num_channel, num_bottleneck[i]))

    def hybrid_forward(self, F, x):
        x = self.backbone(x)  # pylint: disable=not-callable
        x = self.feat_trans(x)
        feat = x
        outs = []
        for i in range(self.num_stage):
            feat, ht, paf = self.cpm[i](feat)
            feat = F.concat(feat, x)
            outs.append([ht, paf])
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
            heatmap_flip = out[-1][0][1].asnumpy().astype('float64')
            paf_flip = out[-1][1][1].asnumpy().astype('float64')
            heatmap, paf = flip_prediction(heatmap, heatmap_flip, paf, paf_flip)
        return heatmap, paf


##### model for version 3

class CPNGlobalBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_limb, num_channel):
        super(CPNGlobalBlock, self).__init__()
        with self.name_scope():
            self.P4 = nn.Conv2D(num_channel, kernel_size=1, activation='relu')   # 1/4
            self.P8 = nn.Conv2D(num_channel, kernel_size=1, activation='relu')   # 1/8
            self.P16 = nn.Conv2D(num_channel, kernel_size=1, activation='relu')  # 1/16
            self.upx2 = UpSamplingBlock(num_channel, 2)
            self.T8 = nn.Conv2D(num_channel, kernel_size=1, activation='relu')  # 1/16 -> 1/8
            self.T4 = nn.Conv2D(num_channel, kernel_size=1, activation='relu')  # 1/8 -> 1/4
            self.Pre4 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.Pre8 = KpsPafBlock(num_kps, num_limb, num_channel)
            self.Pre16 = KpsPafBlock(num_kps, num_limb, num_channel)

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
            self.feat_trans = ConvBnReLU(num_channel)
            self.pred = KpsPafBlock(num_kps, num_limb, num_channel)

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
            self.global_net = CPNGlobalBlock(num_kps, num_limb, num_channel)
            self.refine_net = CPNRefineBlock(num_kps, num_limb, num_channel)

    def hybrid_forward(self, F, x):
        feats = self.backbone(x)  # pylint: disable=not-callable
        f4, f8, f16, ht4, ht8, ht16, paf4, paf8, paf16 = self.global_net(*feats)
        r_ht, r_paf = self.refine_net(f4, f8, f16)
        return (ht4, paf4, ht8, paf8, ht16, paf16), (r_ht, r_paf)

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
        _, out = self(batch)
        heatmap = out[0][0].asnumpy().astype('float64')
        paf = out[1][0].asnumpy().astype('float64')
        if flip:
            heatmap_flip = out[0][1].asnumpy().astype('float64')
            paf_flip = out[1][1].asnumpy().astype('float64')
            heatmap, paf = flip_prediction(heatmap, heatmap_flip, paf, paf_flip)
        return heatmap, paf


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
        super(MaskPoseNet, self).__init__(prefix='maskpose')
        with self.name_scope():
            self.backbone = None
            self.head = MaskHeatHead(num_kps, num_channel)
            self.refine = RefineNet(num_kps, num_channel)

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
        prefix, backbone, num_stage, num_channel, batch_size, optim, epoch = parse_from_name_v2(model)
        net = PoseNet(num_kps=num_kps, num_limb=num_limb, num_stage=num_stage, num_channel=num_channel)
        creator, featname, fixed = cfg.BACKBONE_v2[backbone]
    elif version == 3:
        prefix, backbone, num_channel, batch_size, optim, epoch = parse_from_name_v3(model)
        net = CascadePoseNet(num_kps=num_kps, num_limb=num_limb, num_channel=num_channel)
        creator, featname, fixed = cfg.BACKBONE_v3[backbone]
    elif version == 4:
        prefix, backbone, num_channel, batch_size, optim, epoch = parse_from_name_v3(model)
        net = MaskPoseNet(num_kps=num_kps, num_channel=num_channel)
        creator, featname, fixed = cfg.BACKBONE_v4[backbone]
    else:
        raise RuntimeError('no such version %d'%version)
    net.init_backbone(creator, featname, fixed, pretrained=False)
    net.load_params(model, mx.cpu(0))
    return net


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
