from os import listdir
from xml.dom import minidom
import csv

dir_path = "./NLMCXR_png/"

imgs = []

for filename in listdir(dir_path):
    imgs.append(filename)

img_labels = []
with open('./files/image_labels.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        fname = row['Filename']
        label = row['Label']
        label_text = row['LabelText']
        img_labels.append((fname, label, label_text))

not_found = []
img_labels_fnames = [x[0] for x in img_labels]

for img in imgs:
    if img not in img_labels_fnames:
        not_found.append(img)
#not_found.remove('Thumbs.db')

missing_files = [x[3:7] for x in not_found]

dir_path = "./NLMCXR_reports/"

impressions = []




for filename in listdir(dir_path):
    if filename.endswith('.xml'):
        for ms in missing_files:
            if filename == ms+'.xml':
                mydoc = minidom.parse(dir_path+filename)
                items = mydoc.getElementsByTagName('AbstractText')
                for item in items:
                    if item.attributes['Label'].value == 'IMPRESSION':
                        if item.firstChild is not None:
                            impressions.append((filename[:-4], item.firstChild.data))


need_labels = []


for fname in not_found:
    for tpl in impressions:
        if fname[3:7] == tpl[0]:
            need_labels.append((fname, tpl[1]))

need_labels = list(set(need_labels))

with open('./files/need_labels.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for tpl in need_labels:
        csvWriter.writerow([tpl[0], 2, tpl[1]])

