import os
from os import listdir
from xml.dom import minidom

dir_path = "./NLMCXR_reports/"
impressions = {}
findings = {}
majors = {}

# extract text fields from .xml files

for filename in listdir(dir_path):
    if filename.endswith('.xml'):
      mydoc = minidom.parse(dir_path+filename)
      major = mydoc.getElementsByTagName('major')
      if major[0].firstChild is not None:
        majors.update({filename: major[0].firstChild.data})


for filename in listdir(dir_path):
    if filename.endswith('.xml'):
      mydoc = minidom.parse(dir_path+filename)
      items = mydoc.getElementsByTagName('AbstractText')
      for item in items:
        if item.attributes['Label'].value == 'FINDINGS':
          if item.firstChild is not None:
            findings.update({filename: item.firstChild.data})
        if item.attributes['Label'].value == 'IMPRESSION':
          if item.firstChild is not None:
            impressions.update({filename: item.firstChild.data})


with open('./files/impressions_files.txt', 'w', encoding='utf-8') as w:
  for f, imp in impressions.items():
    w.write(f'{f}, {imp}\n')

with open('./files/findings_files.txt', 'w', encoding='utf-8') as w:
  for f, fin in findings.items():
    w.write(f'{f}, {fin}\n')

with open('./files/majors_files.txt', 'w', encoding='utf-8') as w:
  for f, m in majors.items():
    w.write(f'{f}, {m}\n')

# Map images to their corresponding .xml reports
png = '.png'
images = []

for filename in listdir(dir_path):
    if filename.endswith('.xml'):
      mydoc = minidom.parse(dir_path+filename)
      items = mydoc.getElementsByTagName('parentImage')
      for item in items:
          id = item.attributes['id'].value
          id = id + png
          images.append((filename, id))



with open('./files/reports_to_images.csv', 'w', encoding='utf-8') as f:
    csvWriter = csv.writer(f, delimiter=',',
                          quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Report', 'Image'])
    for tpl in images:
        csvWriter.writerow([tpl[0], tpl[1]])


