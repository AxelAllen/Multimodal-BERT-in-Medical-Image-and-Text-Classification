import csv

with open('./files/normal_updated.txt', 'r', encoding='utf-8') as f:
    normal = f.read().lower().split('\n')


with open('./files/abnormal_updated.txt', 'r', encoding='utf-8') as f:
    abnormal = f.read().lower().split('\n')

normal_csv = []
with open('./files/normal.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        fname = row['xmlId']
        label = row['label']
        label_text = row['label_text']
        normal_csv.append((fname, label, label_text))

abnormal_csv = []
with open('./files/abnormal_extended.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        fname = row['xmlId']
        label = row['label']
        label_text = row['label_text']
        abnormal_csv.append((fname, label, label_text))

does_not_match = {}
counter = 0
for line in normal:
    file = line.partition('\t')[0]
    impr = line.partition('\t')[2]
    for row in normal_csv:
        file_csv = row[0]
        impr_csv = row[2]
        if file == file_csv:
            if impr == impr_csv:
                counter += 1
            else:
                does_not_match.update({file: (impr, impr_csv)})

for line in abnormal:
    file = line.partition('\t')[0]
    impr = line.partition('\t')[2]
    for row in abnormal_csv:
        file_csv = row[0]
        impr_csv = row[2]
        if file == file_csv:
            if impr == impr_csv:
                counter += 1
            else:
                does_not_match.update({file: (impr, impr_csv)})

print(f'{counter} items in normal/abnormal.csv matched {len(normal)+len(abnormal)} items in normal/abnormal.txt')