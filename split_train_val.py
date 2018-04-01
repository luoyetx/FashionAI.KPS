from __future__ import print_function, division

import os
import pandas as pd
from config import cfg


train_csv = os.path.join(cfg.DATA_DIR, 'train/Annotations/train.csv')
df = pd.read_csv(train_csv)
df = df.sample(frac=1, random_state=666).reset_index(drop=True)
train_num = int(len(df) * cfg.TRAIN_RATE)
df_train = df[:train_num]
df_test = df[train_num:]

out = os.path.join(cfg.DATA_DIR, 'train.csv')
df_train.to_csv(out, index=False)
out = os.path.join(cfg.DATA_DIR, 'val.csv')
df_test.to_csv(out, index=False)

print('train.csv')
print(df_train['image_category'].value_counts())
print('val.csv')
print(df_test['image_category'].value_counts())
