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


import argparse

from qualitygui import RetinaQualityLabel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retina Quality Labeler')
    parser.add_argument('-e', '--relabel-mode', required=False, action='store_false', help='Re-labeling mode')
    parser.add_argument('-s', '--image-size', required=False, default=512, type=int, help='Image size')
    parser.add_argument('-o', '--options-file', required=True, help='Options file (json).')
    args = parser.parse_args()

    gui = RetinaQualityLabel(
        edit_mode=args.relabel_mode, image_size=args.image_size, options_file=args.options_file
    )
    gui.init_app()
