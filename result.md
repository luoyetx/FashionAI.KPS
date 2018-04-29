# season 1

### test-a

v1
==

5 model for 5 category

- v1.1 model with ..., use detect_kps_v1 with cheat, public 0.1288

v2
==

joint train

- v2.1 default-vgg19-S5-C64-BS32-adam-0050.params, use detect_kps_v1 without cheat, err on val 0.097577, public 0.1555
- v2.2 same model as v2.1, use detect_kps_v1 with cheat trick,                      err on val 0.098896, public 0.1037
- v2.3 same model as v2.1, use detect_kps_v2 with cheat trick,                      err on val 0.099015, public 0.1043
- v2.4 same model as v2.1, use detect_kps_v2 without cheat trick,                   err on val 0.097711, public 0.1561

fix eval

- v2.2 same model as v2.1, use detect_kps_v1 with cheat trick, heatmap test err 5.601, err on val 0.099024, public 0.1037
- v2.5 V2.more_data-vgg19-S5-C64-BS32-adam-0050.params, heatmap test err 5.356, err on val 0.093222, public 0.0963
- v2.6 based on v2.5 with image flip augment, err on val 0.086405, public 0.0883
- v2.7 V2.more_data_longterm-vgg19-S5-C64-BS32-sgd-0063.params, heatmap test err 5.213, err on val 0.076914, public 0.0805
- v2.8 based on v2.7 using cv2.INTER_CUBIC, err on val 0.070487, public 0.0755
- v2.9 based on v2.8 using paf flip and mutli scale, err on val 0.054828, public 0.0589, missing 6579

more augment, fix augment, center on image not object

- v2.10 V2.default.fresh-vgg19-S5-C64-BS32-sgd-0081.params, heatmap test err 5.043, err on val 0.052433, public 0.0549
- v2.11 V2.default.fresh-vgg19-S5-C64-BS32-sgd-0087.params, heatmap test err 5.026, err on val 0.050738, public 0.0543
- v2.12 V2.default.more_data.fint-vgg19-S5-C64-BS32-sgd-0019.params heatmap test err 4.964, err on val 0.048695, public 0.0546, missing 1334
- v2.13 V2.default.more_data.fint-vgg19-S5-C64-BS32-sgd-0046.params heatmap test err 4.895, err on val 0.048706, public 0.0542, missing 1495

fix scale from [512, 440, 386] to [512, 368, 224]

- v2.14 based on v2.12, err on val 0.045472, public 0.0505, missing 756
- v2.15 based on v2.14 predict missing with similarity trans, err on val 0.045404, public 0.0506, missing 756
- v2.16 based on v2.13, err on val 0.045635, public 0.0499, missing 836

**number of missing points matters**

```python
import pandas as pd
from dataset import FashionAIKPSDataSet

test=pd.read_csv('./result/submission-v2.9/result.csv')
test = FashionAIKPSDataSet(test)

c = 0
for i in range(len(test)):
    cate = test.category[i]
    kps = test.kps[i]
    landmark_idx = cfg.LANDMARK_IDX[cate]
    for j in landmark_idx:
        if kps[j, 2] == 0:
            c += 1
print(c)
```

- v2.17 V2.test-vgg19-S3-C64-BS16-sgd-0087.params test err 5.051566, err on val 0.046370, public 0.0505, missing 747

### test-b

- v2.18 V2.default.more_data.fint-vgg19-S5-C64-BS32-sgd-0019.params heatmap test err 4.964, err on val 0.045472, public 0.0502, missing 834
- v2.19 V2.default.more_data.fint-vgg19-S5-C64-BS32-sgd-0046.params heatmap test err 4.895, err on val 0.045635, public 0.0498, missing 960
- v2.20 V2.test-vgg19-S3-C64-BS16-sgd-0087.params test err 5.051566, err on val 0.046370, public ???, missing 818
- v2.21 V2.test.fine.m29-vgg19-S3-C64-BS16-sgd-0035.params test err 4.983602, err on val 0.044493, public 0.0498, missing 842

sclae exploration

- v2.18 err on val 0.045048, scales to [440, 368, 224]
- v2.19 err on val 0.045607, scales to [440, 368, 224]
- v2.20 err on val 0.046677, scales to [440, 368, 224]
- v2.21 err on val 0.045841, scales to [440, 368, 224]

- v2.21 err on val 0.060549, scale 512
- v2.21 err on val 0.052712, scale 440
- v2.21 err on val 0.048561, scale 400
- v2.21 err on val 0.048460, scale 368
- v2.21 err on val 0.051302, scale 296
- v2.21 err on val 0.055201, scale 224

- v2.22 based on v2.21 scales to [440, 368, 224], err on val 0.045841, public 0.0495, missing 794
- v2.23 based on v2.21 scales to [400, 368, 296], err on val 0.046198, public 0.0503, missing 950

# season 2

### test-a
