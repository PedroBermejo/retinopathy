from os import listdir

from attr import has

path_images = "/Users/pbermejo/Documents/Master/retinopathy-dataset/"
path_good_quality_train = "train/good/"
path_bad_quality_train = "train/bad/"
path_good_quality_val = "val/good/"
path_bad_quality_val = "val/bad/"
path_good_quality_test = "test/good/"
path_bad_quality_test = "test/bad/"

all_images_names = (listdir(path_images + path_good_quality_train) 
    + listdir(path_images + path_good_quality_val)
    + listdir(path_images + path_good_quality_test)
    + listdir(path_images + path_bad_quality_train) 
    + listdir(path_images + path_bad_quality_val)
    + listdir(path_images + path_bad_quality_test))

print("All images size: " + str(len(all_images_names)/2))

# Verify if an image is duplicated
hasDuplicates = False
for imageName in all_images_names:
    if all_images_names.count(imageName) > 1: 
        hasDuplicates = True

print("All images has duplicates: " + str(hasDuplicates))
