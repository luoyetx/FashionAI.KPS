from __future__ import print_function, division

import os
import pandas as pd
from config import cfg

data_dir = cfg.DATA_DIR
# train
train_csv = os.path.join(data_dir, 'train/Annotations/train.csv')
df = pd.read_csv(train_csv)
df['image_id'] = 'train/' + df['image_id']
# warmup
warmup_csv = os.path.join(data_dir, 'train/Annotations/annotations.csv')
df_warmup = pd.read_csv(warmup_csv)
df_warmup['image_id'] = 'train/' + df_warmup['image_id']
# r1-test-a
test_a_csv = os.path.join(data_dir, 'r1-test-a/fashionAI_key_points_test_a_answer_20180426.csv')
df_test_a = pd.read_csv(test_a_csv)
df_test_a['image_id'] = 'r1-test-a/' + df_test_a['image_id']
# r1-test-b
test_b_csv = os.path.join(data_dir, 'r1-test-b/fashionAI_key_points_test_b_answer_20180426.csv')
df_test_b = pd.read_csv(test_b_csv)
df_test_b['image_id'] = 'r1-test-b/' + df_test_b['image_id']

# split train and val
df = pd.concat([df, df_warmup, df_test_a, df_test_b])
df = df.sample(frac=1, random_state=666).reset_index(drop=True)
train_ratio = 0.9
train_num = int(len(df) * train_ratio)
df_train = df[:train_num]
df_test = df[train_num:]
out = os.path.join(data_dir, 'train.csv')
df_train.to_csv(out, index=False)
out = os.path.join(data_dir, 'val.csv')
df_test.to_csv(out, index=False)

print('train.csv')
print('train num:', len(df_train))
print(df_train['image_category'].value_counts())
print('val.csv')
print('val num:', len(df_test))
print(df_test['image_category'].value_counts())
