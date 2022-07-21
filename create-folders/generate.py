from os import listdir, unlink
from os.path import splitext, join, isfile, islink, isdir
import argparse
import random
import shutil
import os
import pandas as pd


def main():
    # this flag will include messidor and idrid datasets, all images are good quality
    # just setting this to true will result in unbalanced dataset
    include_idrid_messidor = True
    # this will include only 35 percent of all good quality images, so adding idrid and
    # messidor won't result in unbalanced dataset
    include_idrid_messidor_balanced = True

    trainRate = 0.70
    valRate = 0.85
    testRate = 1
    balanceRate = 0.35

    JPG_EXT = ".jpg"

    # Empty train and val folders
    folders = [join(os.getcwd(), args.path_good_quality_train),
               join(os.getcwd(), args.path_bad_quality_train),
               join(os.getcwd(), args.path_good_quality_val),
               join(os.getcwd(), args.path_bad_quality_val),
               join(os.getcwd(), args.path_good_quality_test),
               join(os.getcwd(), args.path_bad_quality_test)]

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
    base_csv_DF = pd.read_csv(join(os.getcwd(), args.csv_path))

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
        for imageName in listdir(join(os.getcwd(), args.path_images_idrid)):
            goodQualityIdrid.append(imageName)

        for imageName in listdir(join(os.getcwd(), args.path_images_messidor)):
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
            shutil.copyfile(join(os.getcwd(), args.path_images + name), join(os.getcwd(), args.path_good_quality_train + name))
            imageName = splitext(name)[0] + JPG_EXT
            shutil.copyfile(path_images + imageName, path_images + path_good_quality_train + imageName)
            countGQ_train += 1
        elif rand < valRate:
            shutil.copyfile(join(os.getcwd(), args.path_images + name), join(os.getcwd(), args.path_good_quality_val + name))
            imageName = splitext(name)[0] + JPG_EXT
            shutil.copyfile(join(os.getcwd(), args.path_images + imageName), join(os.getcwd(), args.path_good_quality_val + imageName))
            countGQ_val += 1
        else:
            shutil.copyfile(join(os.getcwd(), args.path_images + name), join(os.getcwd(), args.path_good_quality_test + name))
            imageName = splitext(name)[0] + JPG_EXT
            shutil.copyfile(join(os.getcwd(), args.path_images + imageName), join(os.getcwd(), args.path_good_quality_test + imageName))
            countGQ_test += 1

    for name in badQuality:
        rand = random.random()
        if rand < trainRate:
            shutil.copyfile(join(os.getcwd(), args.path_images + name), join(os.getcwd(), args.path_bad_quality_train + name))
            imageName = splitext(name)[0] + JPG_EXT
            shutil.copyfile(join(os.getcwd(), args.path_images + imageName), join(os.getcwd(), args.path_bad_quality_train + imageName))
            countBQ_train += 1
        elif rand < valRate:
            shutil.copyfile(join(os.getcwd(), args.path_images + name), join(os.getcwd(), args.path_bad_quality_val + name))
            imageName = splitext(name)[0] + JPG_EXT
            shutil.copyfile(join(os.getcwd(), args.path_images + imageName), join(os.getcwd(), args.path_bad_quality_val + imageName))
            countBQ_val += 1
        else:
            shutil.copyfile(join(os.getcwd(), args.path_images + name), join(os.getcwd(), args.path_bad_quality_test + name))
            imageName = splitext(name)[0] + JPG_EXT
            shutil.copyfile(join(os.getcwd(), args.path_images + imageName), join(os.getcwd(), args.path_bad_quality_test + imageName))
            countBQ_test += 1

    # Copy image from idrid and messidor, only good quality images
    if include_idrid_messidor:
        for name in goodQualityIdrid:
            if include_idrid_messidor_balanced:
                if random.random() >= balanceRate:
                    continue
            rand = random.random()
            if rand < trainRate:
                shutil.copyfile(join(os.getcwd(), args.path_images_idrid + name), join(os.getcwd(), args.path_good_quality_train + name))
                countGQ_train += 1
            elif rand < valRate:
                shutil.copyfile(join(os.getcwd(), args.path_images_idrid + name), join(os.getcwd(), args.path_good_quality_val + name))
                countGQ_val += 1
            else:
                shutil.copyfile(join(os.getcwd(), args.path_images_idrid + name), join(os.getcwd(), args.path_good_quality_test + name))
                countGQ_test += 1

        for name in goodQualityMessidor:
            if include_idrid_messidor_balanced:
                if random.random() >= balanceRate:
                    continue
            rand = random.random()
            if rand < trainRate:
                shutil.copyfile(join(os.getcwd(), args.path_images_messidor + name), join(os.getcwd(), args.path_good_quality_train + name))
                countGQ_train += 1
            elif rand < valRate:
                shutil.copyfile(join(os.getcwd(), args.path_images_messidor + name), join(os.getcwd(), args.path_good_quality_val + name))
                countGQ_val += 1
            else:
                shutil.copyfile(join(os.getcwd(), args.path_images_messidor + name), join(os.getcwd(), args.path_good_quality_test + name))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images', help='path for images to select from')
    parser.add_argument('--path-images-idrid', help='path for idrid images')
    parser.add_argument('--path-images-messidor', help='path for messidor images')
    parser.add_argument('--path-good-quality-train', help='path to save good quality train')
    parser.add_argument('--path-bad-quality-train', help='path to save bad quality train')
    parser.add_argument('--path-good-quality-val', help='path to save good quality validation')
    parser.add_argument('--path-bad-quality-val', help='path to save bad quality validation')
    parser.add_argument('--path-good-quality-test', help='path to save good quality test')
    parser.add_argument('--path-bad-quality-test', help='path to save bad quality test')
    parser.add_argument('--csv-path', help='csv where good/bad values are for --path-images')
    args = parser.parse_args()

    main()

