import pandas as pd
from collections import Counter
from natsort import index_natsorted
import numpy as np

ids = []
text = []
ab_ids = []
ab_text = []

normal_vocab_freq_dist = Counter()
ab_vocab_freq_dist = Counter()

# keywords that most likely associated with abnormalities
KEYWORDS = ['emphysema', 'cardiomegaly', 'borderline', 'mild', 'chronic', 'minimal', 'copd', 'hernia',
                 'hyperinflated', 'hemodialysis', 'atelectasis', 'degenerative', 'effusion', 'atherosclerotic',
            'aneurysmal', 'granuloma', 'fracture', 'severe', 'concerns', 'fibrosis', 'scarring', 'crowding', 'opacities',
            'persistent', 'ectatic', 'hyperinflation', 'moderate', 'opacity', 'calcified', 'effusions', 'edema',
            'continued', 'low lung volume', 'pacing lead', 'resection', 'dilated', 'left', 'right', 'bilateral',
            'hyperexpanded', 'calcification', 'concerning', 'concern', 'enlargement', 'lines', 'tubes', 'Emphysema',
            'Hyperexpanded', 'advanced', 'Advanced', 'tortuosity']


with open('normal.txt', mode='r', encoding='utf-8') as f, open('abnormal.txt', mode='r', encoding='utf-8') as af:
    for line in f:
        xml, *label_text = line.split()
        ids.append(xml)
        normal_vocab_freq_dist.update(label_text)
        text.append(' '.join(label_text))

    for line in af:
        xml, *label_text = line.split()
        ab_ids.append(xml)
        ab_vocab_freq_dist.update(label_text)
        ab_text.append(' '.join(label_text))


def first_filter_normal_label(a_string):
    if a_string.startswith(('no acute', 'no evidence', 'no active', 'no radiographic evidence')) and a_string.endswith(
            ('process.', 'disease.', 'abnormality.', 'abnormalities.', 'findings.', 'finding.', 'identified.',
             'infiltrates.', 'infiltrate.')):
        return 0
    else:
        return a_string


def second_filter_normal(a_string):
    if isinstance(a_string, int):
        return a_string
    if a_string.startswith(('normal chest', 'normal exam', 'unremarkable chest', 'unremarkable examination',
                            'unremarkable radiographs')):
        return 0
    if a_string.startswith('clear') and a_string.endswith('lungs.'):
        return 0
    if a_string.startswith(('negative for', 'negative chest')):
        return 0
    if a_string.startswith('negative') and a_string.endswith('negative.'):
        return 0
    else:
        return a_string


def third_filter_normal(a_string):
    if isinstance(a_string, int):
        return a_string
    if a_string.startswith(('stable appearance', 'stable chest radiograph', 'stable exam', 'stable',
                            'stable post-procedural', 'stable radiographic')):
        if any(w in a_string for w in KEYWORDS):
            return a_string
        else:
            return 0
    if a_string.startswith('clear') or a_string.endswith('clear.'):
        if any(w in a_string for w in KEYWORDS):
            return a_string
        else:
            return 0

    return a_string


def fourth_filter_normal(a_string):
    if isinstance(a_string, int):
        return a_string
    if 'no acute' or 'without acute' in a_string:
        if any(w in a_string for w in KEYWORDS):
            return 2
        elif 'stable' or 'clear' or 'normal' in a_string:
            return 0

    return a_string


print(normal_vocab_freq_dist.most_common(50))
print(ab_vocab_freq_dist.most_common(50))

# filtering strickt normal from borderline/mild abnormal e.g. stable/chronic conditions but no acute findings
normal = {'xmlId': ids, 'label_text': text}
normal_df = pd.DataFrame(normal)
normal_df['label'] = normal_df['label_text']
normal_df['label'] = normal_df['label'].apply(first_filter_normal_label)
normal_df['label'] = normal_df['label'].apply(second_filter_normal)
normal_df['label'] = normal_df['label'].apply(third_filter_normal)
normal_df['label'] = normal_df['label'].apply(fourth_filter_normal)

print(normal_df.loc[normal_df['label'] != 0])
print(normal_df.loc[normal_df['label'] == 0])

print('creating data frame from abnormal.txt')
# dataframe for abnormal file
ab_normal = {'xmlId': ab_ids, 'label_text': ab_text}
ab_normal_df = pd.DataFrame(ab_normal)
ab_normal_df['label'] = 1
print(ab_normal_df.head())


# for easy editing if needed later, want label to be the second column
cols = list(normal_df.columns)
a, b = cols.index('label_text'), cols.index('label')
cols[b], cols[a] = cols[a], cols[b]
normal_df = normal_df[cols]
print(normal_df.head())


# same goes for the abnormal one
cols = list(ab_normal_df.columns)
a, b = cols.index('label_text'), cols.index('label')
cols[b], cols[a] = cols[a], cols[b]
ab_normal_df = ab_normal_df[cols]
print(ab_normal_df.head())

# writing to csv
normal_df.to_csv('normal_with_stable.csv', index=False)
ab_normal_df.to_csv('abnormal.csv', index=False)

# consider class 2 as abnormal and add to abnormal, save to csv sorted descending by 'xml id'
normal_strict_df = normal_df.loc[normal_df['label'] == 0]
normal_strict_df = normal_strict_df.sort_values(by='xmlId',
                                                key=lambda x: np.argsort(index_natsorted(normal_strict_df['xmlId'])))
normal_strict_df.to_csv('normal.csv', index=False)

chronic_df = normal_df.loc[normal_df['label'] != 0].copy()
chronic_df['label'] = 1
abnormal_extended_df = pd.concat([chronic_df, ab_normal_df])
abnormal_extended_df = abnormal_extended_df.sort_values(by='xmlId',
                                                        key=lambda x:
                                                        np.argsort(index_natsorted(abnormal_extended_df['xmlId'])))
abnormal_extended_df.to_csv('abnormal_extended.csv', index=False)
print(abnormal_extended_df)
