#!/usr/bin/env bash

# model version 2
python -u train.py --version 2 --batch-size 32 --optim sgd --lr 1e-4 --lr-decay 0.1 --steps 50,80 --epoches 100 --gpu 0,1 --backbone vgg19 --freq 50 --prefix test  --num-stage 5 --num-channel 64

# model version 3
python -u train.py --version 3 --batch-size 32 --optim adam --lr 1e-4 --lr-decay 0.1 --steps 50,80 --epoches 100 --gpu 0,1 --backbone resnet50 --freq 50 --prefix test  --num-channel 64

# model version 4
python -u train.py --version 4 --batch-size 32 --optim adam --lr 1e-4 --lr-decay 0.1 --steps 50,80 --epoches 100 --gpu 0,1 --backbone resnet50 --freq 50 --prefix test --num-channel 64

# model version 5
python -u train.py --version 5 --batch-size 32 --optim sgd --lr 1e-4 --lr-decay 0.1 --steps 50,80 --epoches 100 --gpu 0,1 --backbone vgg19 --freq 50 --prefix test  --num-stage 5 --num-channel 64

# model detection
python -u train_det.py --batch-size 32 --optim sgd --lr 1e-3 --lr-decay 0.1 --steps 5,10 --epoches 15 --gpu 0,1 --backbone resnet50 --freq 50 --prefix test

# prediction
python -u predict.py --model ./output/V2.test-vgg19-S3-C64-BS32-sgd-0001.params --version 2 --gpu 0 --num-worker 3 --multi-scale --type val --show

# detection prediction
python -u predict_det.py --det-model ./output/Det.default-resnet50-BS32-sgd-0015.params --kps-model ./output/V2.test-vgg19-S3-C64-BS32-sgd-0082.params --version 2 --gpu 0 --num-worker 3 --multi-scale --type val --show
