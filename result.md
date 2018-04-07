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
- v2.9 based on v2.8 using paf flip and mutli scale, err on val 0.054828, public 0.0589
- v2.10

v3
==

Cascade Pose
