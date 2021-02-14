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



labels = []
with open('./files/labels_updated.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        xml_id = row['xmlId']
        label = row['label']
        label_text = row['label_text']
        labels.append((xml_id, label, label_text))

findings = {}
with open('./files/all_text_files.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    for entry in data:
        fname = entry.partition(',')[0]
        fin = entry.partition(',')[2]
        findings.update({fname: fin})


f_labels = []
for tpl in labels:
    try:
        fin = findings[tpl[0]]
        f_labels.append((tpl[0], tpl[1], fin))
    except KeyError:
        continue

img_labels = []

for ltpl in f_labels:
    xml_id = ltpl[0]
    for rtpl in rep_to_img:
        if rtpl[0] == xml_id:
            img_labels.append((rtpl[1], ltpl[1], ltpl[2]))


labels_frontal = []

for (fname, label, label_t) in img_labels:
    if fname in imgs:
        labels_frontal.append((fname, label, label_t))

with open('./files/image_labels_both.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for tpl in img_labels:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

with open('./files/image_labels_both_frontal.csv', 'w', encoding='utf-8') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for tpl in labels_frontal:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])
