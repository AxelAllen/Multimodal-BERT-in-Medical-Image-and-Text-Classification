import csv


rep_to_img = []

with open('./files/reports_to_images.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        report = row['Report']
        img = row['Image']
        rep_to_img.append((report, img))


## For impression label
labels = []
with open('./files/labels_updated.csv', 'r') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        xml_id = row['xmlId']
        label = row['label']
        label_text = row['label_text']
        labels.append((xml_id, label, label_text))

img_labels = []

for ltpl in labels:
    xml_id = ltpl[0]
    for rtpl in rep_to_img:
        if rtpl[0] == xml_id:
            img_labels.append((rtpl[1], ltpl[1], ltpl[2]))

with open('./files/image_labels.csv', 'w', encoding='utf-8') as csv_file:
    csvWriter = csv.writer(csv_file, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['Filename', 'Label', 'LabelText'])
    for tpl in img_labels:
        csvWriter.writerow([tpl[0], tpl[1], tpl[2]])
