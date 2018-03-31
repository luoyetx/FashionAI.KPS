#!/usr/bin/env bash

datadir=./data
category=skirt
batchsize=16
optim=adam
lr=1e-4
wd=1e-5
epoches=100
cpmstages=5
cpmchannels=64
backbone=vgg19
freq=20
seed=666
steps=30,60
gpu=0
log=./log/$category-$backbone-S$cpmstages-C$cpmchannels-$optim.log

python -u train.py --data-dir $datadir --category $category --batch-size $batchsize --optim $optim --lr $lr --epoches $epoches --cpm-stages $cpmstages --cpm-channels $cpmchannels --backbone $backbone --freq $freq --gpu $gpu --seed $seed --steps $steps 2>&1 | tee $log
