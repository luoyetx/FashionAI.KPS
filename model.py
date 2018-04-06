from __future__ import print_function, division

import os
import numpy as np
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn

from config import cfg
from utils import process_cv_img


def freeze_bn(block):
    if isinstance(block, nn.BatchNorm):
        print('freeze batchnorm operator', block.name)
        block._kwargs['use_global_stats'] = True
        block.gamma.grad_req = 'null'
        block.beta.grad_req = 'null'
    else:
        for child in block._children:
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
        heatmap = out[-1][0][0].asnumpy()
        paf = out[-1][1][0].asnumpy()
        if flip:
            heatmap_flip = out[-1][0][1].asnumpy()
            heatmap_flip = heatmap_flip[:, :, ::-1]
            for i, j in cfg.LANDMARK_SWAP:
                tmp = heatmap_flip[i].copy()
                heatmap_flip[i] = heatmap_flip[j]
                heatmap_flip[j] = tmp
            heatmap = (heatmap + heatmap_flip) / 2
        return heatmap, paf



class CPNGlobalBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_channel=64):
        super(CPNGlobalBlock, self).__init__()
        self.num_kps = num_kps
        self.num_channel = num_channel
        with self.name_scope():
            self.P4 = self.block()   # 1/4
            self.P8 = self.block()   # 1/8
            self.P16 = self.block()  # 1/16
            self.T8 = nn.Conv2D(self.num_kps, 3, 1, 1)  # 1/16 -> 1/8
            self.T4 = nn.Conv2D(self.num_kps, 3, 1, 1)  # 1/8 -> 1/4

    def hybrid_forward(self, F, x4, x8, x16):
        F4, F8, F16 = x4, x8, x16
        # 1/16
        R16 = self.P16(F16)
        # 1/8
        U8 = F.UpSampling(self.T8(R16), scale=2, sample_type='nearest')
        R8 = self.P8(F8)
        U8 = F.Crop(U8, R8)
        R8 = R8 + U8
        # 1/4
        U4 = F.UpSampling(self.T4(R8), scale=2, sample_type='nearest')
        R4 = self.P4(F4)
        U4 = F.Crop(U4, R4)
        R4 = R4 + U4
        return R4, R8, R16

    def block(self):
        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(self.num_channel, 3, 1, 1, activation='relu'))
            net.add(nn.Conv2D(self.num_kps, 3, 1, 1))
        return net


class CPNRefineBlock(gl.HybridBlock):

    def __init__(self, num_kps, num_channel):
        super(CPNRefineBlock, self).__init__()
        self.num_kps = num_kps
        self.num_channel = num_channel
        with self.name_scope():
            self.P16 = nn.HybridSequential()
            self.P16.add(self.bottleneck(), self.bottleneck(), self.bottleneck())
            self.P8 = nn.HybridSequential()
            self.P8.add(self.bottleneck(), self.bottleneck())
            self.P4 = nn.HybridSequential()
            self.P4.add(self.bottleneck())
            self.R = nn.Conv2D(self.num_kps, 3, 1, 1)

    def hybrid_forward(self, F, x4, x8, x16):
        F4, F8, F16 = x4, x8, x16
        F4 = self.P4(F4)
        F8 = self.P8(F8)
        F16 = self.P16(F16)
        U16 = F.UpSampling(F16, scale=4, sample_type='nearest')
        U16 = F.Crop(U16, F4)
        U8 = F.UpSampling(F8, scale=2, sample_type='nearest')
        U8 = F.Crop(U8, F4)
        out = self.R(F.concat(F4, U8, U16))
        return out

    def bottleneck(self):
        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(self.num_channel, 3, 1, 1, activation='relu'))
            net.add(nn.Conv2D(self.num_channel, 3, 1, 1, activation='relu'))
        return net


class CascadePoseNet(gl.HybridBlock):

    def __init__(self, num_kps, num_channel):
        super(CascadePoseNet, self).__init__()
        self.num_kps = num_kps
        self.num_channel = num_channel
        with self.name_scope():
            self.backbone = None
            self.global_net = CPNGlobalBlock(self.num_kps, self.num_channel)
            self.refine_net = CPNRefineBlock(self.num_kps, self.num_channel)

    def hybrid_forward(self, F, x):
        feats = self.backbone(x)  # pylint: disable=not-callable
        global_pred = self.global_net(*feats)
        refine_pred = self.refine_net(*global_pred)
        return global_pred, refine_pred

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
        global_pred, refine_pred = self(batch)
        heatmap = refine_pred[0].asnumpy()
        if flip:
            heatmap_flip = refine_pred[1].asnumpy()
            heatmap_flip = heatmap_flip[:, :, ::-1]
            for i, j in cfg.LANDMARK_SWAP:
                tmp = heatmap_flip[i].copy()
                heatmap_flip[i] = heatmap_flip[j]
                heatmap_flip[j] = tmp
            heatmap = (heatmap + heatmap_flip) / 2
        return heatmap



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
    else:
        prefix, backbone, num_channel, batch_size, optim, epoch = parse_from_name_v3(model)
        net = CascadePoseNet(num_kps=num_kps, num_channel=num_channel)
        creator, featname, fixed = cfg.BACKBONE_v3[backbone]
    net.init_backbone(creator, featname, fixed, pretrained=False)
    net.load_params(model, mx.cpu(0))
    return net
