from __future__ import print_function, division

import numpy as np
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn

from config import cfg
from utils import process_cv_img, parse_from_name


def freeze_bn(block):
    if isinstance(block, nn.BatchNorm):
        print('freeze batchnorm operator', block.name)
        block._kwargs['use_global_stats'] = True
        block.gamma.grad_req = 'null'
        block.beta.grad_req = 'null'
    else:
        for child in block._children:
            freeze_bn(child)


def install_backbone(net, creator, featnames, fixed):
    with net.name_scope():
        backbone = creator(pretrained=True)
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

    def __init__(self, num_output, channels, ks=[3, 3, 3, 1, 1], **kwargs):
        super(CPMBlock, self).__init__(**kwargs)
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

    def __init__(self, num_kps, num_limb, stages, channels, **kwargs):
        super(PoseNet, self).__init__(**kwargs)
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

    def init_backbone(self, creator, featname, fixed):
        install_backbone(self, creator, [featname], fixed)

    def predict(self, img, ctx):
        data = process_cv_img(img)
        batch = mx.nd.array(data[np.newaxis], ctx)
        out = self(batch)
        heatmap = out[-1][0][0].asnumpy()
        paf = out[-1][1][0].asnumpy()
        return heatmap, paf


def load_model(model):
    num_kps = cfg.NUM_LANDMARK
    num_limb = len(cfg.PAF_LANDMARK_PAIR)
    prefix, backbone, cpm_stages, cpm_channels, batch_size, optim, epoch = parse_from_name(model)
    net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
    creator, featname, fixed = cfg.BACKBONE[backbone]
    net.init_backbone(creator, featname, fixed)
    net.load_params(model, mx.cpu(0))
    return net


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

    def hybrid_forward(self, F, x):
        F4, F8, F16 = x
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
        self.num_kps = num_kps
        self.num_channel = num_channel
        pass

    def block(self):
        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(self.num_channel, 3, 1, 1, activation='relu'))
            net.add()
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
        global_pred = self.global_net(feats)
        refine_pred = self.refine_net(global_pred)
        return global_pred, refine_pred

    def init_backbone(self, creator, featnames, fixed):
        assert len(featnames) == 3
        install_backbone(self, creator, featnames, fixed)
