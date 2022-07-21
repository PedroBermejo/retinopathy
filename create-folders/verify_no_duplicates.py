import argparse
import os


def main():
    path_good_quality_train = "train/good/"
    path_bad_quality_train = "train/bad/"
    path_good_quality_val = "val/good/"
    path_bad_quality_val = "val/bad/"
    path_good_quality_test = "test/good/"
    path_bad_quality_test = "test/bad/"

    all_images_names = (os.listdir(os.path.join(os.getcwd(), args.base_path, path_good_quality_train))
        + os.listdir(os.path.join(os.getcwd(), args.base_path, path_good_quality_val))
        + os.listdir(os.path.join(os.getcwd(), args.base_path, path_good_quality_test))
        + os.listdir(os.path.join(os.getcwd(), args.base_path, path_bad_quality_train))
        + os.listdir(os.path.join(os.getcwd(), args.base_path, path_bad_quality_val))
        + os.listdir(os.path.join(os.getcwd(), args.base_path, path_bad_quality_test)))

    print("All images size: " + str(len(all_images_names)/2))

    # Verify if an image is duplicated
    hasDuplicates = False
    for imageName in all_images_names:
        if all_images_names.count(imageName) > 1:
            hasDuplicates = True

    print("All images has duplicates: " + str(hasDuplicates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', help='path for images')
    args = parser.parse_args()

    main()
