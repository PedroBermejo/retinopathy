import matplotlib.pyplot as plt
from os import listdir
from os.path import splitext
import re
import csv

path_r_labels = "/Users/pedro_bermejo/Epam-OneDrive/OneDrive - EPAM/Maestria/retinopatia-dataset/labels.csv"
path_copy = "/Users/pedro_bermejo/Epam-OneDrive/OneDrive - EPAM/Maestria/retinopatia-dataset/labeled-copy"
path_relaxed = "/Users/pedro_bermejo/Epam-OneDrive/OneDrive - EPAM/Maestria/retinopatia-dataset/labeled-relaxed"
path_restricted = "/Users/pedro_bermejo/Epam-OneDrive/OneDrive - EPAM/Maestria/retinopatia-dataset/labeled-restricted"

x = []
titleArray = [0, 0, 0, 0, 0]

imageNames = [
    name for name in listdir(path_restricted)
    if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
]

for name in imageNames:
    with open(path_r_labels, "r") as infile:
        reader = csv.reader(infile)
        #next(reader)
        find = splitext(name)[0]
        for line in reader:
            if  find == line[0]:
                x.append(line[1])
                titleArray[int(line[1])] += 1
                break
                
title = '0: ' + str(titleArray[0]) + \
    ', 1: ' + str(titleArray[1]) + \
    ', 2: ' + str(titleArray[2]) + \
    ', 3: ' + str(titleArray[3]) + \
    ', 4: ' + str(titleArray[4]) + \
    ', Total: ' + str(len(x))

print(len(x))
plt.title(title)
plt.hist(x)
plt.show()