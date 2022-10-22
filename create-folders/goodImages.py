from os import listdir, unlink
from os.path import splitext, join, isfile, islink, isdir
import argparse
import random
import shutil
import os
import pandas as pd


def main():
    # Empty existing albumentation folders
    for folder in listdir(join(os.getcwd(), args.path_save_difference)):
        folder_path = join(os.getcwd(), args.path_save_difference, folder)
        if isdir(folder_path):
            # print(folder_path)
            for filename in listdir(folder_path):
                file_path = join(os.getcwd(), args.path_save_difference, folder, filename)
                try:
                    # shutil.rmtree(file_path)
                    os.remove(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    good_images = { }
    folders = ['train/good', 'val/good', 'test/good']

    for folder in folders:
        for image_name in listdir(join(os.getcwd(), args.path_all_images, folder)):
            if image_name != '.DS_Store' and not image_name.endswith('.json'):
                good_images[image_name] = join(os.getcwd(), args.path_all_images, folder, image_name)

    print("Length all images:", len(good_images))

    for folder in folders:
        for image_name in listdir(join(os.getcwd(), args.path_used_images, folder)):
            if good_images.get(image_name):
                good_images.pop(image_name)

    print("Length difference images:", len(good_images))

    for image_name, src_path in good_images.items():
        shutil.copyfile(src_path, join(os.getcwd(), args.path_save_difference, 'good_sample', image_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-all-images', help='path for all images to select from')
    parser.add_argument('--path-used-images', help='path for images already used to train')
    parser.add_argument('--path-save-difference', help='path to save the difference')
    args = parser.parse_args()

    main()
