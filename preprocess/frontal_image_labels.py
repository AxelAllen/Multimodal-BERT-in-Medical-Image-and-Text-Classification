import csv
from os import listdir

dir_path = 'NLMCXR_png_frontal'

imgs = []
for filename in listdir(dir_path):
    if filename.endswith('.png'):
        imgs.append(filename)

labels = []
with open('./files/image_labels.csv', 'r', encoding='utf-8') as csvfile:
    csvReader = csv.DictReader(csvfile)
    for row in csvReader:
        filename = row['Filename']
        label = row['Label']
        label_text = row['LabelText']
        labels.append((filename, label, label_text))

labels_frontal = []

for (fname, label, label_t) in labels:
    if fname in imgs:
        labels_frontal.append((fname, label, label_t))


with open('./files/image_labels_frontal.csv', 'w', encoding='utf-8') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for tpl in labels_frontal:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])