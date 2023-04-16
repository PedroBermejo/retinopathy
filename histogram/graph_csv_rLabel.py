import argparse
import re
import os
import json

from os import listdir


def main():

    json_paths = []
    folder_counter = {}
    total_images = 0

    for dir in listdir(os.path.join(os.getcwd(), args.path_datasets)):
        if dir != '.DS_Store' and '.png' not in dir:
            for subdir in listdir(os.path.join(os.getcwd(), args.path_datasets, dir)):
                if subdir != '.DS_Store':
                    for name in listdir(os.path.join(os.getcwd(), args.path_datasets, dir, subdir)):
                        if re.match(r'[\w,\d]+\.[json|JSON]{4}', name):
                            json_paths.append(os.path.join(os.getcwd(), args.path_datasets, dir, subdir, name))
                        if re.match(r'[\w,\d]+\.[jpg|png]{3}', name):
                            total_images = total_images + 1
                            if folder_counter.get(dir + '_' + subdir):
                                folder_counter[dir + '_' + subdir] = folder_counter.get(dir + '_' + subdir) + 1
                            else:
                                folder_counter[dir + '_' + subdir] = 1

    label_counter = {}
    total_json = 0

    for json_name in json_paths:
        total_json = total_json + 1
        with open(json_name, 'r') as f:
            data = json.load(f)
            for key in data.keys():
                if label_counter.get(key):
                    label_counter[key] = label_counter.get(key) + int(data[key])
                else:
                    label_counter[key] = int(data[key])

    print("Total json files: ", total_json)
    print("Labels: ", label_counter)
    print("Total images: ", total_images)
    print("Folders: ", folder_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-datasets', help='Path to datasets')
    args = parser.parse_args()
    main()

