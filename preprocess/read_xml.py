import os
from os import listdir
from xml.dom import minidom

dir_path = "./NLMCXR_reports/"
impressions = {}
findings = {}
both = {}


for filename in listdir(dir_path):
    text = ''
    if filename.endswith('.xml'):
      mydoc = minidom.parse(dir_path+filename)
      items = mydoc.getElementsByTagName('AbstractText')
      for item in items:
        if item.attributes['Label'].value == 'FINDINGS':
          if item.firstChild is not None:
            findings.update({filename: item.firstChild.data})
            text += item.firstChild.data
        if item.attributes['Label'].value == 'IMPRESSION':
          if item.firstChild is not None:
            impressions.update({filename: item.firstChild.data})
            text += ' '+item.firstChild.data
    both.update({filename: text})


with open('./files/impressions_files.txt', 'w', encoding='utf-8') as w:
  for f, imp in impressions.items():
    w.write(f'{f}, {imp}\n')

with open('./files/findings_files.txt', 'w', encoding='utf-8') as w:
  for f, fin in findings.items():
    w.write(f'{f}, {fin}\n')

with open('./files/all_text_files.txt', 'w', encoding='utf-8') as w:
  for f, txt in both.items():
    w.write(f'{f}, {txt}\n')

