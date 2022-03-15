from os import listdir, unlink
from os.path import splitext, join, isfile, islink, isdir
import random
import json
import shutil
import pandas as pd

path_images = "/Users/pbermejo/Documents/Master/images/"
path_good_quality_train = "../retinopathy-dataset/train/good/"
path_bad_quality_train = "../retinopathy-dataset/train/bad/"
path_good_quality_val = "../retinopathy-dataset/val/good/"
path_bad_quality_val = "../retinopathy-dataset/val/bad/"
path_good_quality_test = "../retinopathy-dataset/test/good/"
path_bad_quality_test = "../retinopathy-dataset/test/bad/"
csv_path = "/Users/pbermejo/Documents/Master/repos/retinopathy/histogram/result.csv"

trainRate = 0.70
valRate = 0.85
testRate = 1

JPG_EXT = ".jpg"

# Empty train and val folders
folders = [path_images + path_good_quality_train, 
           path_images + path_bad_quality_train, 
           path_images + path_good_quality_val, 
           path_images + path_bad_quality_val,
           path_images + path_good_quality_test,
           path_images + path_bad_quality_test]

# Empty folders first
for folder in folders:
    for filename in listdir(folder):
        file_path = join(folder, filename)
        try:
            if isfile(file_path) or islink(file_path):
                unlink(file_path)
            elif isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# Get all image names from csv
base_csv_DF = pd.read_csv(csv_path)

print(base_csv_DF.head())
print(base_csv_DF.shape)

goodQuality = []
badQuality = []

# Number 1 means bad quality, 0 means good quality
for index, row in base_csv_DF.iterrows():
    number = row['abraham'] + row['ulises'] + row['pedro']
    if number >= 2:
        badQuality.append(row['file_name'])
    else:
        goodQuality.append(row['file_name'])

print("Len good quality: ", len(goodQuality))
print("Len bad quality: ", len(badQuality))

countGQ_train, countBQ_train, countGQ_val, countBQ_val, countGQ_test, countBQ_test = 0, 0, 0, 0, 0, 0

# Copy image and json to train, val and test folders
for name in goodQuality:
    rand = random.random()
    if rand < trainRate:
        shutil.copyfile(path_images + name, path_images + path_good_quality_train + name)
        imageName = splitext(name)[0] + JPG_EXT
        shutil.copyfile(path_images + imageName, path_images + path_good_quality_train + imageName)
        countGQ_train += 1
    elif rand < valRate:
        shutil.copyfile(path_images + name, path_images + path_good_quality_val + name)
        imageName = splitext(name)[0] + JPG_EXT
        shutil.copyfile(path_images + imageName, path_images + path_good_quality_val + imageName)
        countGQ_val += 1
    else:
        shutil.copyfile(path_images + name, path_images + path_good_quality_test + name)
        imageName = splitext(name)[0] + JPG_EXT
        shutil.copyfile(path_images + imageName, path_images + path_good_quality_test + imageName)
        countGQ_test += 1

for name in badQuality:
    rand = random.random()
    if rand < trainRate:
        shutil.copyfile(path_images + name, path_images + path_bad_quality_train + name)
        imageName = splitext(name)[0] + JPG_EXT
        shutil.copyfile(path_images + imageName, path_images + path_bad_quality_train + imageName)
        countBQ_train += 1
    elif rand < valRate:
        shutil.copyfile(path_images + name, path_images + path_bad_quality_val + name)
        imageName = splitext(name)[0] + JPG_EXT
        shutil.copyfile(path_images + imageName, path_images + path_bad_quality_val + imageName)
        countBQ_val += 1
    else:
        shutil.copyfile(path_images + name, path_images + path_bad_quality_test + name)
        imageName = splitext(name)[0] + JPG_EXT
        shutil.copyfile(path_images + imageName, path_images + path_bad_quality_test + imageName)
        countBQ_test += 1


print("Total Good Quality:", len(goodQuality))
print("Total Bad Quality:", len(badQuality))

print("Train Good Quality:", countGQ_train, " {:.2f}".format(100*countGQ_train/len(goodQuality)), "%")
print("Train Bad Quality:", countBQ_train, " {:.2f}".format(100*countBQ_train/len(badQuality)), "%")
print("Validation Good Quality:", countGQ_val, " {:.2f}".format(100*countGQ_val/len(goodQuality)), "%")
print("Validation Bad Quality:", countBQ_val, " {:.2f}".format(100*countBQ_val/len(badQuality)), "%")
print("Test Good Quality:", countGQ_test, " {:.2f}".format(100*countGQ_test/len(goodQuality)), "%")
print("Test Bad Quality:", countBQ_test, " {:.2f}".format(100*countBQ_test/len(badQuality)), "%")
