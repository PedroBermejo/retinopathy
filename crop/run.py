import argparse
from cropper import Cropper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cropper.')
    parser.add_argument('--model_path', required=True, help="Model file name (ckpt).")
    parser.add_argument('--src', required=True, help="Source directory.")
    parser.add_argument('--dst', required=True, help="Destination directory.")
    args = parser.parse_args()

    src_list = args.src.split(',')
    dst_list = args.dst.split(',')
    # print(src_list)
    # print(dst_list)

    for i in range(len(src_list)):
        cropper = Cropper(args.model_path, src_list[i], dst_list[i])
        cropper.crop()


