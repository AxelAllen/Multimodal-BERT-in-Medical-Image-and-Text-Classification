import os
from os import listdir
from xml.dom import minidom
import csv

with open('./files/normal_updated.txt', 'r', encoding='utf-8') as f:
    normal = f.read().lower().split('\n')


with open('./files/abnormal_updated.txt', 'r', encoding='utf-8') as f:
    abnormal = f.read().lower().split('\n')

dir_path = "./NLMCXR_reports/"
impressions = {}

for filename in listdir(dir_path):
    if filename.endswith('.xml'):
      mydoc = minidom.parse(dir_path+filename)
      items = mydoc.getElementsByTagName('AbstractText')
      for item in items:
        if item.attributes['Label'].value == 'IMPRESSION':
          if item.firstChild is not None:
            impressions.update({filename: str(item.firstChild.data).lower()})
            #impressions.append(item.firstChild.data)
          else:
            print(f'{filename} firstChild is None')

counter = 0
for line in normal:
    file = line.partition('\t')[0]
    impr = line.partition('\t')[2]
    for key, value in impressions.items():
        if key == file and impr == value:
            counter += 1

for line in abnormal:
    file = line.partition('\t')[0]
    impr = line.partition('\t')[2]
    for key, value in impressions.items():
        if key == file and impr == value:
            counter += 1

print(f'{counter} items in normal/abnormal.txt matched {len(impressions)} files in /NLMCXR_reports')

normal = []
with open('./files/normal.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        fname = row['xmlId']
        label = row['label']
        label_text = row['label_text']
        normal.append((fname, label, label_text))

abnormal = []
with open('./files/abnormal_extended.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        fname = row['xmlId']
        label = row['label']
        label_text = row['label_text']
        abnormal.append((fname, label, label_text))


counter = 0
for line in normal:
    file = line[0]
    impr = line[2]
    for key, value in impressions.items():
        if key == file:
            if impr == value:
                counter += 1
            else:
                print(f'{file}:\n {value}\n {impr}')

for line in abnormal:
    file = line[0]
    impr = line[2]
    for key, value in impressions.items():
        if key == file:
            if impr == value:
                counter += 1
            else:
                print(f'{file}:\n {value}\n {impr}')

print(f'{counter} items in normal/abnormal.csv matched {len(impressions)} files in /NLMCXR_reports')