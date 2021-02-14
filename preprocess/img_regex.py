import re
from os import listdir
import shutil

dir_path = './NLMCXR_png/'
new_path = './NLMCXR_png_frontal/'

imgs = []
for filename in listdir(dir_path):
    if filename.endswith('.png'):
        file = filename.partition('.')[0]
        imgs.append(file)


pattern = re.compile(r'CXR[\d]+_IM-[\d]+-1001[\d-]*')

matches = []

for img in imgs:
    match = pattern.search(img)
    if match is not None:
        matches.append(match.group())

for filename in matches:
    shutil.copyfile(dir_path+filename+'.png', new_path+filename+'.png')