import csv

with open('./files/majors_files.txt', 'r', encoding='utf-8') as r:
    data = r.read().lower().split('\n')

    files = [s.partition(', ')[0] for s in data]
    majors = [s.partition(', ')[2] for s in data]

labels = []
for m in majors:
    if m == 'normal':
        labels.append(0)
    else:
        labels.append(1)


key_words = ['device', 'technical', 'instruments', 'foreign', 'no indexing']
multi_labels = []
for m in majors:
    kw_check = any(kw in m for kw in key_words)
    if m == 'normal':
        multi_labels.append(0)
    elif kw_check:
        multi_labels.append(2)
    else:
        multi_labels.append(1)



with open('./files/major_labels.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for f, l, m in zip(files, labels, majors):
        csvWriter.writerow([f, l, m])

with open('./files/major_multi_labels.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for f, l, m in zip(files, multi_labels, majors):
        csvWriter.writerow([f, l, m])

