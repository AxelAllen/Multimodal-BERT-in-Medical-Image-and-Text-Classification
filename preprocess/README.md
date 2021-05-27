# Labels

0 = normal with stable, 3214 samples. 
1 = abnormal, 708 samples. 
2 = borderline/stable chronic conditions, 811 samples. 

## The following files are sorted by 'xmlId' column
- normal.csv
- abnormal_extended.csv

0 = normal, 2403 samples. 
1 = abnormal extended, 1519 samples. 

## Steps:
1. read_xml.py and map_report_to_image.py input: xml file  --> normal and abnormal.txt (aka normal_updated, abnormal_updated) 
2. read_impression.py and preprocess.py input: abnormal/normal txt files --> normal.csv, abnormal_extended.csv, .csv files
3. check_labels.py verifies previous steps are done correctly compare.py missing labels.py
4. create image labels from either impression or findings? 
5. 
