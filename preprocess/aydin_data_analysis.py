import pandas as pd
from collections import Counter
from natsort import index_natsorted
import numpy as np


df = pd.read_csv('ids_raw_texts_labels.csv')
print(df.head())
print(df.columns)
df_normal = df.loc[df['Labels'] == 0].copy()
df_abnormal = df.loc[df['Labels'] == 1].copy()
print(f'Total Aydin data frame size: {len(df.index)}')
print(f'number of normal samples: {len(df_normal.index)}')
print(f'number of abnormal samples: {len(df_abnormal.index)}')
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head(10))
