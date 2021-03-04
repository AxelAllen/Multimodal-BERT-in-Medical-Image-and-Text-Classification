import csv

# getting label from major
labels_major = {}
with open('./files/image_labels_major_frontal.csv', 'r', encoding='utf-8') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        img_file_id = row['img']
        label = row['label']
        labels_major[img_file_id] = label

# getting multi label from major
multi_labels_major = {}
with open('./files/image_labels_multi_major_frontal.csv', 'r', encoding='utf-8') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        img_file_id = row['img']
        label = row['label']
        multi_labels_major[img_file_id] = label

# make img, label, label text list of tuples with 2 classes for label and text from impression
labels_major_impression = []
multi_labels_major_impression = []
with open('../data/image_labels_csv/image_labels_impression_frontal.csv', 'r', encoding='utf-8') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        img_file_id = row['Filename']
        label_text = row['LabelText']
        try:
            label = labels_major[img_file_id]
        except KeyError:
            label = row['Label']
        finally:
            labels_major_impression.append((img_file_id, label, label_text))

        try:
            label = multi_labels_major[img_file_id]
        except KeyError:
            label = row['Label']
        finally:
            multi_labels_major_impression.append((img_file_id, label, label_text))

# make img, label, label text list of tuples with 2 classes for label and text from findings
findings_text_frontal = []
multi_labels_major_findings = []
with open('../data/image_labels_csv/image_labels_findings_frontal.csv', 'r', encoding='utf-8') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        img_file_id = row['Filename']
        label_text = row['LabelText']
        try:
            label = labels_major[img_file_id]
        except KeyError:
            label = row['Label']
        finally:
            findings_text_frontal.append((img_file_id, label, label_text))

        try:
            label = multi_labels_major[img_file_id]
        except KeyError:
            label = row['Label']
        finally:
            multi_labels_major_findings.append((img_file_id, label, label_text))

# make img, label, label text list of tuples with 2 classes for label and text from both findings and impression
both_text_frontal = []
multi_labels_major_both = []
with open('../data/image_labels_csv/image_labels_both_frontal.csv', 'r', encoding='utf-8') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        img_file_id = row['Filename']
        label_text = row['LabelText']
        try:
            label = labels_major[img_file_id]
        except KeyError:
            label = row['Label']
        finally:
            both_text_frontal.append((img_file_id, label, label_text))

        try:
            label = multi_labels_major[img_file_id]
        except KeyError:
            label = row['Label']
        finally:
            multi_labels_major_both.append((img_file_id, label, label_text))

# write csv file with img, label, text from impression
with open('../data/image_labels_csv/image_labels_major_impression_frontal.csv', 'w', encoding='utf-8') as csvfile, \
        open('../data/image_labels_csv/image_multi_labels_major_impression_frontal.csv', 'w', encoding='utf-8') as \
                csvfile_multi:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])

    csvWriter_multi = csv.writer(csvfile_multi, delimiter=',',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter_multi.writerow(['Filename', 'Label', 'LabelText'])

    for tpl in labels_major_impression:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

    for tpl in multi_labels_major_impression:
        csvWriter_multi.writerow([tpl[0], tpl[1], tpl[2]])

# write csv file with img, label, text from findings
with open('../data/image_labels_csv/image_labels_major_findings_frontal.csv', 'w', encoding='utf-8') as csvfile, \
        open('../data/image_labels_csv/image_multi_labels_major_findings_frontal.csv', 'w', encoding='utf-8') as \
                csvfile_multi:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])

    csvWriter_multi = csv.writer(csvfile_multi, delimiter=',',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter_multi.writerow(['Filename', 'Label', 'LabelText'])

    for tpl in findings_text_frontal:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

    for tpl in multi_labels_major_findings:
        csvWriter_multi.writerow([tpl[0], tpl[1], tpl[2]])

# write csv file with img, label, text from both
with open('../data/image_labels_csv/image_labels_major_both_frontal.csv', 'w', encoding='utf-8') as csvfile, \
        open('../data/image_labels_csv/image_multi_labels_major_both_frontal.csv', 'w', encoding='utf-8') as \
                csvfile_multi:
    csvWriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])

    csvWriter_multi = csv.writer(csvfile_multi, delimiter=',',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter_multi.writerow(['Filename', 'Label', 'LabelText'])

    for tpl in both_text_frontal:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])

    for tpl in multi_labels_major_both:
        csvWriter_multi.writerow([tpl[0], tpl[1], tpl[2]])
