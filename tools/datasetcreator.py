#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = ["Abraham Sanchez"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco"
__credits__ = ["Abraham Sanchez"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["Abraham Sanchez"]
__email__ = "abraham.sanchez@jalisco.gob.mx"
__status__ = "Development"

import re
import os
import json
import shutil
import argparse

from os import listdir
from os.path import join
from common import create_directory


class DatasetCreator:

    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path

    def create(self):
        images = self.get_images()
        labels = self.get_labels()
        self.split(labels=labels, images=images)

    def get_images(self):
        images = [
            image for image in listdir(self.src_path)
            if re.match(r'[\w,\d]+\.[jpg|JPG|png|PNG|tif|TIF|jpeg|JPEG]{3,4}', image)
        ]
        images_names = {os.path.splitext(x)[0]: x for x in images}
        return images_names

    def get_labels(self):
        labels = [
            image for image in listdir(self.src_path)
            if re.match(r'[\w,\d]+\.[json|JSON]{4}', image)
        ]
        labels_ids = {os.path.splitext(x)[0]: x for x in labels}
        return labels_ids

    def split(self, labels, images):
        for label_id, label_name in labels.items():
            image_name = images.get(label_id)
            if image_name:
                file = open(join(self.src_path, label_name))
                content = json.load(file)
                target = list(content.values())[0]
                dst = join(self.dst_path, str(target))
                src = join(self.src_path, image_name)
                create_directory(dst)
                shutil.copy(src=src, dst=join(dst, image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retina Quality Labeler')
    parser.add_argument('-s', '--source-path', required=True,  help='Images & labels directory.')
    parser.add_argument('-d', '--dest-path', required=True, help='Output directory.')
    args = parser.parse_args()

    creator = DatasetCreator(
        src_path=args.source_path,
        dst_path=args.dest_path
    )
    creator.create()
