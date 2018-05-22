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

```
train.csv
train num: 57717
blouse      12775
skirt       12516
trousers    11721
outwear     10502
dress       10203
Name: image_category, dtype: int64
val.csv
val num: 6414
skirt       1425
blouse      1331
trousers    1325
outwear     1222
dress       1111
Name: image_category, dtype: int64
```

multi-scale [440, 368, 224], flip

- v2.24 V2.default-vgg19-S3-C64-BS32-sgd-0100.params, test err 4.774

```
Average Error for blouse: 0.041192
Average Error for skirt: 0.037188
Average Error for outwear: 0.043027
Average Error for dress: 0.042556
Average Error for trousers: 0.044173
Total Average Error 0.042031
```

public 0.0411

- v2.25 V2.default-vgg19-S5-C64-BS32-sgd-0100.params, test err 4.767

```
Average Error for blouse: 0.040719
Average Error for skirt: 0.035972
Average Error for outwear: 0.043104
Average Error for dress: 0.042869
Average Error for trousers: 0.043175
Total Average Error 0.041723
```

public 0.0403

- v2.26 V2.split.s3-vgg19-S5-C64-BS32-sgd-0100.params, test err 4.718

```
Average Error for blouse: 0.040602
Average Error for skirt: 0.037055
Average Error for outwear: 0.041627
Average Error for dress: 0.042484
Average Error for trousers: 0.042901
Total Average Error 0.041319
```

public 0.0399

- v2.27 ensemble model of v2.25 and v2.26

```
Average Error for blouse: 0.040579
Average Error for skirt: 0.036159
Average Error for outwear: 0.041778
Average Error for dress: 0.042013
Average Error for trousers: 0.043519
Total Average Error 0.041244
```

public 0.0396

- v2.28 ensemble result of v2.25 and v2.26, skirt from v2.25, others from v2.26

```
Average Error for blouse: 0.040602
Average Error for skirt: 0.035972
Average Error for outwear: 0.041627
Average Error for dress: 0.042484
Average Error for trousers: 0.042901
Total Average Error 0.041215
```

public 0.0399

- v10.0 based on Det.default-resnet50-BS32-sgd-0030.params and V2.default-vgg19-S3-C64-BS32-sgd-0100.params

```
Average Error for blouse: 0.055427
Average Error for skirt: 0.056346
Average Error for outwear: 0.060014
Average Error for dress: 0.047308
Average Error for trousers: 0.056334
Total Average Error 0.054724
```

public 0.0519

- v2.29 V2.split.all.fms-vgg19-S5-C64-BS32-sgd-0093.params, test err 4.689

```
Average Error for blouse: 0.040975
Average Error for skirt: 0.036438
Average Error for outwear: 0.041878
Average Error for dress: 0.042377
Average Error for trousers: 0.043366
Total Average Error 0.041466
```

public 0.0403

- v3.0 V3.default-resnet50-C64-BS32-adam-0045.params, test err on refine 21.13

```
Average Error for blouse: 0.078661
Average Error for skirt: 0.065999
Average Error for outwear: 0.076792
Average Error for dress: 0.066081
Average Error for trousers: 0.065844
Total Average Error 0.071932
```

public ?

more train data, remove ht_all

```
train.csv
train num: 60924
blouse      13423
skirt       13249
trousers    12372
outwear     11103
dress       10777
Name: image_category, dtype: int64
val.csv
val num: 3207
skirt       692
blouse      683
trousers    674
outwear     621
dress       537
Name: image_category, dtype: int64
```

- v2.29 V2.split.all.fms-vgg19-S5-C64-BS32-sgd-0093.params, test err 4.689

```
Average Error for blouse: 0.041580
Average Error for skirt: 0.034749
Average Error for outwear: 0.041892
Average Error for dress: 0.041630
Average Error for trousers: 0.041655
Total Average Error 0.041050
```

public 0.0403

- v2.30 V2.test-vgg19-S3-C64-BS32-sgd-0100.params, test err 2.158

```
Average Error for blouse: 0.040746
Average Error for skirt: 0.036821
Average Error for outwear: 0.042141
Average Error for dress: 0.041398
Average Error for trousers: 0.044314
Total Average Error 0.041431
```

public 0.0407

- v3.1 V3.default-resnet50-C64-BS16-adam-0060.params, test err on refine 8.221

```
Average Error for blouse: 0.064561
Average Error for skirt: 0.048746
Average Error for outwear: 0.063303
Average Error for dress: 0.054035
Average Error for trousers: 0.057418
Total Average Error 0.059184
```

public 0.0554

- v2.31 V2.mexpo-resnet50-S3-C256-BS24-adam-0055.params, no mask to feat, test err 2.086

```
Average Error for blouse: 0.037986
Average Error for skirt: 0.031357
Average Error for outwear: 0.043468
Average Error for dress: 0.042158
Average Error for trousers: 0.038983
Total Average Error 0.039812
```

public 0.0397

- v2.32 V2.mexpo-resnet50-S3-C256-BS24-adam-0070.params, no mask to feat, test err 2.086

```
Average Error for blouse: 0.037898
Average Error for skirt: 0.031830
Average Error for outwear: 0.043459
Average Error for dress: 0.041509
Average Error for trousers: 0.038677
Total Average Error 0.039627
```

public 0.0395

- v2.33 V2.default-resnet50-S3-C256-BS28-adam-0067.params, test err 2.071945

```
Average Error for blouse: 0.038148
Average Error for skirt: 0.032480
Average Error for outwear: 0.042623
Average Error for dress: 0.041367
Average Error for trousers: 0.040135
Total Average Error 0.039756
```

public 0.0394

- v2.34 V2.no.occ-resnet50-S3-C256-C2-BS28-adam-0070.params, num_context=2, test err 1.378

```
Average Error for blouse: 0.037351
Average Error for skirt: 0.032302
Average Error for outwear: 0.041242
Average Error for dress: 0.041117
Average Error for trousers: 0.038042
Total Average Error 0.038805
```

public 0.0388

- v2.35 V2.no.occ-resnet50-S3-C256-C2-BS28-adam-0065.params, num_context=2, test err 1.377

```
Average Error for blouse: 0.037600
Average Error for skirt: 0.032327
Average Error for outwear: 0.041127
Average Error for dress: 0.041228
Average Error for trousers: 0.037826
Total Average Error 0.038841
```

public 0.0388

- v2.36 V2.context3-resnet50-S3-C256-C3-BS24-adam-0070.params, test err 1.339

```
Average Error for 0: 0.027584
Average Error for 1: 0.028100
Average Error for 2: 0.028959
Average Error for 3: 0.030314
Average Error for 4: 0.031895
Average Error for 5: 0.055038
Average Error for 6: 0.058547
Average Error for 7: 0.053436
Average Error for 8: 0.057075
Average Error for 9: 0.037626
Average Error for 10: 0.039014
Average Error for 11: 0.044090
Average Error for 12: 0.044049
Average Error for 13: 0.038228
Average Error for 14: 0.039460
Average Error for 15: 0.026199
Average Error for 16: 0.025881
Average Error for 17: 0.034650
Average Error for 18: 0.032838
Average Error for 19: 0.063997
Average Error for 20: 0.036305
Average Error for 21: 0.033759
Average Error for 22: 0.037222
Average Error for 23: 0.031461
Average Error for blouse: 0.036703
Average Error for skirt: 0.028923
Average Error for outwear: 0.039063
Average Error for dress: 0.039211
Average Error for trousers: 0.037146
Total Average Error 0.037209
```

public 0.0381

- v3.1 V3.test-resnet50-C256-BS18-adam-0031.params

```
G-h-04 = 5.503272
R-h-04 = 5.410425
G-h-08 = 1.586563
R-h-08 = 1.385941
G-h-16 = 0.487300
R-h-16 = 0.433932
G-p-04 = 106.700837
R-p-04 = 104.853014
G-p-08 = 41.696519
R-p-08 = 37.243585
G-p-16 = 19.133359
R-p-16 = 15.260324
```

```
Average Error for 0: 0.027609
Average Error for 1: 0.028415
Average Error for 2: 0.029838
Average Error for 3: 0.031052
Average Error for 4: 0.031315
Average Error for 5: 0.056688
Average Error for 6: 0.059660
Average Error for 7: 0.055281
Average Error for 8: 0.058379
Average Error for 9: 0.036900
Average Error for 10: 0.036786
Average Error for 11: 0.042185
Average Error for 12: 0.039063
Average Error for 13: 0.036154
Average Error for 14: 0.040996
Average Error for 15: 0.025382
Average Error for 16: 0.024123
Average Error for 17: 0.035997
Average Error for 18: 0.037289
Average Error for 19: 0.065110
Average Error for 20: 0.038688
Average Error for 21: 0.036447
Average Error for 22: 0.040637
Average Error for 23: 0.036677
Average Error for blouse: 0.036655
Average Error for skirt: 0.029832
Average Error for outwear: 0.039259
Average Error for dress: 0.037934
Average Error for trousers: 0.039136
Total Average Error 0.037335
```

public 0.0371

- v3.2 V3.test-resnet50-C256-BS18-adam-0041.params

```
G-h-04 = 5.315247
R-h-04 = 5.224319
G-h-08 = 1.521610
R-h-08 = 1.338649
G-h-16 = 0.464119
R-h-16 = 0.414197
G-p-04 = 104.350060
R-p-04 = 102.596077
G-p-08 = 40.705234
R-p-08 = 36.354065
G-p-16 = 18.629567
R-p-16 = 14.907007
```

```
Average Error for 0: 0.026701
Average Error for 1: 0.027429
Average Error for 2: 0.029056
Average Error for 3: 0.030637
Average Error for 4: 0.030788
Average Error for 5: 0.055129
Average Error for 6: 0.058358
Average Error for 7: 0.054421
Average Error for 8: 0.056657
Average Error for 9: 0.035908
Average Error for 10: 0.035950
Average Error for 11: 0.041294
Average Error for 12: 0.037208
Average Error for 13: 0.034756
Average Error for 14: 0.037447
Average Error for 15: 0.025070
Average Error for 16: 0.024479
Average Error for 17: 0.034003
Average Error for 18: 0.035534
Average Error for 19: 0.064829
Average Error for 20: 0.033768
Average Error for 21: 0.037104
Average Error for 22: 0.040102
Average Error for 23: 0.034142
Average Error for blouse: 0.035472
Average Error for skirt: 0.029095
Average Error for outwear: 0.037487
Average Error for dress: 0.037153
Average Error for trousers: 0.037899
Total Average Error 0.036142
```

public 0.0364


- tmp

5.244 -> 0.036178 -> 0.0362
5.241 -> 0.036167 -> 0.0363
5.270 -> 0.036001 -> 0.0363

ohkm.13.dev2
4.693 -> 0.035788 -> 0.0362
