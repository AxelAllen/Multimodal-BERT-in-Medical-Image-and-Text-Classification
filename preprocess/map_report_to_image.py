import os
from os import listdir
from xml.dom import minidom
import csv

dir_path = "./NLMCXR_reports/"
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
