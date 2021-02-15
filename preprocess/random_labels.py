import csv
import numpy as np
from os import listdir
from os.path import isfile, join


def writeToCSV(filename, file_path):
    try:
        # filename, 'w/wb'
        with open(filename, 'w') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow(['Filename', 'Label'])
            img_filenames = [f for f in listdir(file_path) if isfile(join(file_path, f))]
            for name in img_filenames:
                csvWriter.writerow([name, int(np.random.randint(2))])
    except OSError as os:
        print("Error: {}".format(os))


writeToCSV('./files/rand_labels.csv', './NLMCXR_png/')
