from os import listdir
from os.path import splitext
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json

images_path = "/Users/pbermejo/Documents/Master/images"

imageNames = [
    name for name in listdir(images_path)
    if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
]

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

test_data_melted = pd.melt(resultDF, id_vars='Categoría', var_name="Valor", value_name="Imagenes")
sns.barplot(x='Categoría', y="Imagenes", hue="Valor", data=test_data_melted)

plt.show()

