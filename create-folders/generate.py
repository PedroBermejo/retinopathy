from os import listdir, unlink
from os.path import splitext, join, isfile, islink, isdir
import random
import json
import shutil
import pandas as pd

# this flag will include messidor and idrid datasets, all images are good quality
# just setting this to true will result in unbalanced dataset
include_idrid_messidor = True
# this will include only 35 percent of all good quality images, so adding idrid and
# messidor won't result in unbalanced dataset
include_idrid_messidor_balanced = True

path_images = "/Users/pbermejo/Documents/Master/images/"
path_good_quality_train = "../retinopathy-dataset/train/good/"
path_bad_quality_train = "../retinopathy-dataset/train/bad/"
path_good_quality_val = "../retinopathy-dataset/val/good/"
path_bad_quality_val = "../retinopathy-dataset/val/bad/"
path_good_quality_test = "../retinopathy-dataset/test/good/"
path_bad_quality_test = "../retinopathy-dataset/test/bad/"
csv_path = "/Users/pbermejo/Documents/Master/repos/retinopathy/histogram/result.csv"

path_images_idrid = "/Users/pbermejo/Documents/Master/idrid-crop/idrid_crop/"
path_images_messidor = "/Users/pbermejo/Documents/Master/messidor-crop/messidor_crop/"

trainRate = 0.70
valRate = 0.85
testRate = 1
balanceRate = 0.35

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
goodQualityIdrid = []
goodQualityMessidor = []

# Number 1 means bad quality, 0 means good quality
for index, row in base_csv_DF.iterrows():
    number = row['abraham'] + row['ulises'] + row['pedro']
    if number >= 2:
        badQuality.append(row['file_name'])
    else:
        goodQuality.append(row['file_name'])

# append idrid and messidor datasets based on flag
if include_idrid_messidor:
    for imageName in listdir(path_images_idrid):
        goodQualityIdrid.append(imageName)
        
    for imageName in listdir(path_images_messidor):
        goodQualityMessidor.append(imageName)

print("Len good quality: ", len(goodQuality))
print("Len bad quality: ", len(badQuality))
print("Len good quality Idrid: ", len(goodQualityIdrid))
print("Len good quality Messidor: ", len(goodQualityMessidor))

countGQ_train, countBQ_train, countGQ_val, countBQ_val, countGQ_test, countBQ_test = 0, 0, 0, 0, 0, 0

# Copy image and json to train, val and test folders
for name in goodQuality:
    if include_idrid_messidor_balanced:
        if random.random() >= balanceRate:
            continue
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

# Copy image from idrid and messidor, only good quality images
if include_idrid_messidor:
    for name in goodQualityIdrid:
        if include_idrid_messidor_balanced:
            if random.random() >= balanceRate:
                continue
        rand = random.random()
        if rand < trainRate:
            shutil.copyfile(path_images_idrid + name, path_images + path_good_quality_train + name)
            countGQ_train += 1
        elif rand < valRate:
            shutil.copyfile(path_images_idrid + name, path_images + path_good_quality_val + name)
            countGQ_val += 1
        else:
            shutil.copyfile(path_images_idrid + name, path_images + path_good_quality_test + name)
            countGQ_test += 1

    for name in goodQualityMessidor:
        if include_idrid_messidor_balanced:
            if random.random() >= balanceRate:
                continue
        rand = random.random()
        if rand < trainRate:
            shutil.copyfile(path_images_messidor + name, path_images + path_good_quality_train + name)
            countGQ_train += 1
        elif rand < valRate:
            shutil.copyfile(path_images_messidor + name, path_images + path_good_quality_val + name)
            countGQ_val += 1
        else:
            shutil.copyfile(path_images_messidor + name, path_images + path_good_quality_test + name)
            countGQ_test += 1

totalGoodQualityLen = len(goodQuality) + len(goodQualityIdrid) + len(goodQualityMessidor)

if include_idrid_messidor_balanced:
    newTotalGoodQualityLen = countGQ_train + countGQ_val + countGQ_test
    print("Total Good Quality:", newTotalGoodQualityLen)
    print("Total Bad Quality:", len(badQuality))

    print("Train Good Quality:", countGQ_train, " {:.2f}".format(100*countGQ_train/newTotalGoodQualityLen), "%")
    print("Train Bad Quality:", countBQ_train, " {:.2f}".format(100*countBQ_train/len(badQuality)), "%")
    print("Validation Good Quality:", countGQ_val, " {:.2f}".format(100*countGQ_val/newTotalGoodQualityLen), "%")
    print("Validation Bad Quality:", countBQ_val, " {:.2f}".format(100*countBQ_val/len(badQuality)), "%")
    print("Test Good Quality:", countGQ_test, " {:.2f}".format(100*countGQ_test/newTotalGoodQualityLen), "%")
    print("Test Bad Quality:", countBQ_test, " {:.2f}".format(100*countBQ_test/len(badQuality)), "%")
else:
    print("Total Good Quality:", totalGoodQualityLen)
    print("Total Bad Quality:", len(badQuality))

    print("Train Good Quality:", countGQ_train, " {:.2f}".format(100*countGQ_train/totalGoodQualityLen), "%")
    print("Train Bad Quality:", countBQ_train, " {:.2f}".format(100*countBQ_train/len(badQuality)), "%")
    print("Validation Good Quality:", countGQ_val, " {:.2f}".format(100*countGQ_val/totalGoodQualityLen), "%")
    print("Validation Bad Quality:", countBQ_val, " {:.2f}".format(100*countBQ_val/len(badQuality)), "%")
    print("Test Good Quality:", countGQ_test, " {:.2f}".format(100*countGQ_test/totalGoodQualityLen), "%")
    print("Test Bad Quality:", countBQ_test, " {:.2f}".format(100*countBQ_test/len(badQuality)), "%")
