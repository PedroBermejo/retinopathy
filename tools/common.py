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


def create_directory(directory_name):
    """
    Create folder where data is going to be saved
    """
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    except IOError:
        raise
