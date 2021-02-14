import pandas as pd
from collections import Counter
from natsort import index_natsorted
import numpy as np
from preprocess import *


# label the need_labels.csv file to 0 or 1
df = pd.read_csv('files/need_labels.csv')
df['Label'] = df['LabelText']
print(df.head())

df['Label'] = df['Label'].apply(first_filter_normal_label)
df['Label'] = df['Label'].apply(second_filter_normal)
df['Label'] = df['Label'].apply(third_filter_normal)
df['Label'] = df['Label'].apply(fourth_filter_normal)


print(df.loc[df['Label'] != 0])
print(df.loc[df['Label'] == 0])

patch_abnormal_df = df.loc[df['Label'] != 0].copy()
patch_abnormal_df['Label'] = 1

patch_normal_df = df.loc[df['Label'] == 0].copy()

patch_normal_df.to_csv('files/patch_normal.csv', index=False)
patch_abnormal_df.to_csv('files/patch_abnormal.csv', index=False)
