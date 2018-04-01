from __future__ import print_function, division

import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn


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
        feat = self.backbone(x)
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
        with self.name_scope():
            backbone = creator(pretrained=True)
            data = mx.sym.var('data')
            out_name = '_'.join([backbone.name, featname, 'fwd_output'])
            out = backbone(data).get_internals()[out_name]
            name = backbone.name
            self.backbone = gl.SymbolBlock(out, data, params=backbone.collect_params())
            # hacking parameters
            params = self.backbone.collect_params()
            for key, item in params.items():
                should_fix = False
                for pattern in fixed:
                    if name + '_' + pattern + '_' in key:
                        should_fix = True
                if should_fix:
                    print('fix', key)
                    item.grad_req = 'null'
            # special for batchnorm

