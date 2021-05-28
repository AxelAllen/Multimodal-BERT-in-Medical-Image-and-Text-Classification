# Labels

0 = normal with stable, 3214 samples.   
1 = abnormal, 708 samples.   
2 = borderline/stable chronic conditions, 811 samples.   

## The following files are sorted by 'xmlId' column
- normal.csv
- abnormal_extended.csv

0 = normal, 2403 samples.   
1 = abnormal extended, 1519 samples.   

## Steps to run the script in this directory:
1. `read_xml.py` reads the .xml report files and generate text files and map reports to images.
2. `preprocess_impressions.py` and `preprocess_major.py` preprocess the text to generate labels based on the 'impression' or 'major' metadata.
3. `check_labels.py` check for empty text fields and discard samples that are missing text data.
4. `filter_frontal_images.py` select filter for frontal images and move them to a separate directory
5. `get_image_labels*.py` scripts generates .csv dataset files with various labeling schemes.

Files in the **files/** directory can be generated from the raw data files by running the scripts in this directory.
