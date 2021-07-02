#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = ["Abraham Sanchez"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco"
__credits__ = ["Abraham Sanchez"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["Abraham Sanchez"]
__email__ = ["abraham.sanchez@jalisco.gob.mx"]
__status__ = "Development"


import os
import json
import numpy as np
import argparse

from os.path import splitext
from os.path import join
from PIL import Image
from PIL import ImageDraw

from common import create_directory


class MaskCreator:
    """
    Create a labeled image that contains the anatomical elements of a fundus photography.
    """

    def __init__(self, annot_path, output_path):
        """
        Constructor.
        :param annot_path: str, the annotation directory.
        :param output_path: str, the output directory.
        """
        self.annot_path = annot_path
        self.output_path = output_path
        create_directory(directory_name=output_path)

    def create_masks(self):
        """
        Create and save images that contain both optic disc (value 1) and macula (value 2).
        """
        for root, directories, files in os.walk(self.annot_path):
            for file_name in files:
                annotations = open(join(root, file_name))
                data = json.load(annotations)
                # Extract attributes
                image_height = data['imageHeight']
                image_width = data['imageWidth']
                shapes = data['shapes']
                # Create empty image
                image = Image.new(mode='L', size=(image_height, image_width), color=0)
                draw = ImageDraw.Draw(image)
                # Create the optic disc and the macula
                for shape in shapes:
                    label = shape['label']
                    points = shape['points']
                    if label == 'disc':
                        draw = self.draw_optic_disc(draw=draw, points=points)
                    elif label == 'macula':
                        draw = self.draw_macula(draw=draw, points=points)
                # Save file as tif
                image_name, _ = splitext(file_name)
                image.save(join(self.output_path, image_name + '.tif'))

    @classmethod
    def draw_optic_disc(cls, draw, points):
        """
        Draw the shape of the optic disc on the image.
        :param draw: PIL draw, draw object.
        :param points: list, list of points (polygons).
        :return: A draw object.
        """
        points = tuple(map(tuple, points))
        draw.polygon(xy=points, fill=1)
        return draw

    @classmethod
    def draw_macula(cls, draw, points):
        """
        Draw the shape of the macula on the image.
        :param draw: PIL draw, draw object.
        :param points: list, list of points (Circle).
        :return: A draw object.
        """
        points = tuple(map(tuple, points))
        x0, y0 = np.array(points[0])
        x1, y1 = np.array(points[1])
        p1 = np.array([x0, y0])
        p2 = np.array([x1, y1])
        r = np.linalg.norm(p2 - p1)
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=2)
        return draw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masks Image Creator.')
    parser.add_argument('-a', '--annot_path', required=True, help='Annotations directory')
    parser.add_argument('-o', '--output_path', required=True, help='Output directory')
    args = parser.parse_args()

    creator = MaskCreator(annot_path=args.annot_path, output_path=args.output_path)
    creator.create_masks()
