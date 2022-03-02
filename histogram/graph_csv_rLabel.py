from os import listdir
from os.path import splitext
import re
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json


labels_path = "/Users/pedro_bermejo/Epam-OneDrive/OneDrive - EPAM/Maestria/retinopatia-dataset/labels.csv"
images_path = "/Users/pedro_bermejo/Epam-OneDrive/OneDrive - EPAM/Maestria/retinopatia-dataset/labeled-restricted"

xValues = []
titleArray = [0, 0, 0, 0, 0]

imageNames = [
    name for name in listdir(images_path)
    if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
]

for name in imageNames:
    with open(labels_path, "r") as infile:
        reader = csv.reader(infile)
        #next(reader)
        find = splitext(name)[0]
        for line in reader:
            if  find == line[0]:
                xValues.append(line[1])
                titleArray[int(line[1])] += 1
                break
                
title = '0: ' + str(titleArray[0]) + \
    ', 1: ' + str(titleArray[1]) + \
    ', 2: ' + str(titleArray[2]) + \
    ', 3: ' + str(titleArray[3]) + \
    ', 4: ' + str(titleArray[4]) + \
    ', Total: ' + str(len(xValues))

bin_values, bin_labels = np.histogram(xValues, [0, 1, 2, 3, 4, 5])

print(title)
print(len(xValues))
print(bin_values)
print(bin_labels)

all_files = glob.glob(images_path + '/*.json')

listDF = []

for filename in all_files:
    with open(filename) as f:
        data = json.load(f)
        df = pd.DataFrame(data, index=[0])
        listDF.append(df)

frame = pd.concat(listDF, ignore_index=True)

print(frame.head())
print(frame.shape)

groupFrameZ = (frame == 0).sum(axis=0)
groupFrameO = (frame == 1).sum(axis=0)

resultDF = pd.concat([pd.DataFrame(groupFrameZ), 
    pd.DataFrame(groupFrameO).rename(columns={0: 1})], axis=1)

resultDF['Categoría'] = resultDF.index

print(resultDF)

sns.barplot(x=[0, 1, 2, 3, 4], y=bin_values)
plt.title(title)
plt.show()

test_data_melted = pd.melt(resultDF, id_vars='Categoría', var_name="Valor", value_name="Imagenes")
sns.barplot(x='Categoría', y="Imagenes", hue="Valor", data=test_data_melted)

plt.show()

