# single v3.1
python -u predict.py --model ./output/final/v3/V3.fine.59.scale3.ohkm-resnet50-C256-BS16-adam-0027.params --version 3 --scale 3 --num-worker 3 --gpu 0 --multi-scale --type test --prefix single.v3.1

# single v3.2
python -u predict.py --model ./output/final/v3/V3.fine.59.scale3.ohkm-resnet50-C256-BS16-adam-0013.params --version 3 --scale 3 --num-worker 3 --gpu 0 --multi-scale --type test --prefix single.v3.2

# single v4.1
python -u predict.py --model ./output/final/v4/V4.noatten.noohkm-resnet50-C256-BS36-adam-0070.params --version 4 --scale 0 --num-worker 3 --gpu 0 --multi-scale --type test --prefix single.v4.1

# single v4.2
python -u predict.py --model ./output/final/v4/V4.noatten.ohkm-resnet50-C256-BS18-adam-0020.params --version 4 --scale 0 --num-worker 3 --gpu 0 --multi-scale --type test --prefix single.v4.2

# ensemble v3 and v4.1
python -u predict.py --model ./output/final/v3/V3.fine.59.scale3.ohkm-resnet50-C256-BS16-adam-0027.params --version 3 --scale 3 --emodel ./output/final/v4/V4.noatten.noohkm-resnet50-C256-BS36-adam-0070.params --eversion 4 --escale 0 --num-worker 3 --gpu 0 --multi-scale --type test --prefix ensemble1

# ensemble v3 and v4.2
python -u predict.py --model ./output/final/v3/V3.fine.59.scale3.ohkm-resnet50-C256-BS16-adam-0027.params --version 3 --scale 3 --emodel ./output/final/v4/V4.noatten.ohkm-resnet50-C256-BS18-adam-0020.params --eversion 4 --escale 0 --num-worker 3 --gpu 0 --multi-scale --type test --prefix ensemble2
