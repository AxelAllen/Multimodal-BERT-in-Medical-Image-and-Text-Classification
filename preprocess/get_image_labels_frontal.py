import csv
from os import listdir

dir_path = 'NLMCXR_png_frontal'

imgs = []
for filename in listdir(dir_path):
    if filename.endswith('.png'):
        imgs.append(filename)

rep_to_img = []

with open('./files/reports_to_images.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        report = row['Report']
        img = row['Image']
        rep_to_img.append((report, img))


## Impression Labels ##

## Extract corresponding labels / text

labels_impr = []
with open('./files/image_labels.csv', 'r', encoding='utf-8') as csvfile:
    csvReader = csv.DictReader(csvfile)
    for row in csvReader:
        filename = row['Filename']
        label = row['Label']
        label_text = row['LabelText']
        labels_impr.append((filename, label, label_text))

labels_frontal_impr = []

for (fname, label, label_t) in labels_impr:
    if fname in imgs:
        labels_frontal_impr.append((fname, label, label_t))


## Save to a file
with open('./files/image_labels_frontal.csv', 'w', encoding='utf-8') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for tpl in labels_frontal_impr:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

## Major Labels ##


## Extract corresponding labels / text
labels_major = []
with open('./files/major_labels.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        xml_id = row['Filename']
        label = row['Label']
        label_text = row['LabelText']
        labels_major.append((xml_id, label, label_text))


multi_labels = []
with open('./files/major_multi_labels.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        xml_id = row['Filename']
        label = row['Label']
        label_text = row['LabelText']
        multi_labels.append((xml_id, label, label_text))


img_labels = []

for tpl in labels_major:
    xml_id = tpl[0]
    for rtpl in rep_to_img:
        if rtpl[0] == xml_id:
            img_labels.append((rtpl[1], tpl[1], tpl[2]))


labels_frontal_major = []

for (fname, label, label_t) in img_labels:
    if fname in imgs:
        labels_frontal_major.append((fname, label, label_t))


img_multi_labels = []

for tpl in multi_labels:
    xml_id = tpl[0]
    for rtpl in rep_to_img:
        if rtpl[0] == xml_id:
            img_multi_labels.append((rtpl[1], tpl[1], tpl[2]))


multi_labels_frontal = []

for (fname, label, label_t) in img_multi_labels:
    if fname in imgs:
        multi_labels_frontal.append((fname, label, label_t))


## Save to separate files
with open('./files/image_labels_major.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['img', 'label', 'text'])
    for tpl in img_labels:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

with open('./files/image_labels_major_frontal.csv', 'w', encoding='utf-8') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['img', 'label', 'text'])
    for tpl in labels_frontal_major:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

with open('./files/image_labels_multi_major.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['img', 'label', 'text'])
    for tpl in img_multi_labels:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

with open('./files/image_labels_multi_major_frontal.csv', 'w', encoding='utf-8') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['img', 'label', 'text'])
    for tpl in multi_labels_frontal:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])