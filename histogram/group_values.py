import os
import re
import pandas as pd
import json
import math
import numpy as np

images_path = "/Users/pbermejo/Documents/Master/images/"
csv_path = "/Users/pbermejo/Documents/Master/repos/retinopathy/histogram/"

# read csv with base evaluation
base_csv = pd.read_csv(csv_path + "reference_working.csv")

print(base_csv.head())
print(base_csv.shape)

imageNames = [
    name for name in os.listdir(images_path)
    if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
]

json_list_DF = []

for name in imageNames:
    with open(images_path + name) as f:
        data = json.load(f)
        df = pd.DataFrame(data, index=[0])
        ev_general = str(df.iloc[0]['Evaluacion general']).split('.')[0]
        base_csv.loc[base_csv['file_name'] == name, 'Pedro'] = ev_general

print(base_csv.head())
print(base_csv.shape)

#base_csv.to_csv(csv_path + "result.csv")