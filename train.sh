#!/usr/bin/env bash

datadir=./data
category=skirt
batchsize=16
optim=sgd
lr=1e-5
wd=1e-5
epoches=100
cpmstages=5
cpmchannels=128
backbone=vgg19
freq=20
seed=666
gpu=1
log=./log/$category-$backbone-S$cpmstages-C$cpmchannels-$optim.log

python -u train.py --data-dir $datadir --category $category --batch-size $batchsize --optim $optim --lr $lr --epoches $epoches --cpm-stages $cpmstages --cpm-channels $cpmchannels --backbone $backbone --freq $freq --gpu $gpu --seed $seed 2>&1 | tee $log
