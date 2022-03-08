from os import listdir, unlink
from os.path import splitext, join, isfile, islink, isdir

import random
import re
import json
import shutil

path_images = "/Users/pbermejo/Documents/Master/images/"
path_good_quality_train = "../retinopathy-dataset/train/good/"
path_bad_quality_train = "../retinopathy-dataset/train/bad/"
path_good_quality_val = "../retinopathy-dataset/val/good/"
path_bad_quality_val = "../retinopathy-dataset/val/bad/"
path_good_quality_test = "../retinopathy-dataset/test/good/"
path_bad_quality_test = "../retinopathy-dataset/test/bad/"

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


# Get all json files from folder
imageNames = [
    name for name in listdir(path_images)
    if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
]

goodQuality = []
badQuality = []

# Split into good a bad qualities
for name in imageNames:
    with open(path_images + name) as f:
        data = json.load(f)
        if data['Evaluacion general'] == 0:
            goodQuality.append(name)
        else: 
            badQuality.append(name)

# print(len(goodQuality))
# print(len(badQuality))

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