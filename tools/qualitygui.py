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


import os
import json

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

from os import listdir
from os.path import join
from os.path import splitext
from tkinter import messagebox as mb
from constants import Components


class RetinaQualityLabel:

    def __init__(self, edit_mode, image_size, options_file):
        self.root = Tk()
        self.root.title(Components.MAIN_FRAME_TITLE)
        self.root_directory = filedialog.askdirectory()
        # Get file names with the most common image extensions
        self.images = [
            image for image in listdir(self.root_directory)
            if re.match(r'[\w,\d]+\.[jpg|JPG|png|PNG|tif|TIF|jpeg|JPEG]{3,4}', image)
        ]
        # Get labeled file names
        self.already_labeled = [
            image.split('.')[0] + '.' for image in listdir(self.root_directory)
            if re.match(r'[\w,\d]+\.[json|JSON]{4}', image)
        ]
        if edit_mode:  # Allow to edit the images that were already labeled
            self.__discard_images()
        # Init evaluations to zero for each image
        self.image_index = 0
        self.component_gui_position = 0
        self.image_size = (image_size, image_size)
        try:
            self.options_components = json.load(open(options_file))
            self.option_actions = [IntVar() for _ in range(len(self.options_components))]
            self.images_scores = {
                image: dict.fromkeys([
                    option['text'] for _, option in self.options_components.items()
                ], 0)
                for image in self.images
            }
        except Exception as e:
            mb.showwarning('Yes', 'Incorrect options file!')
            raise ValueError(e)
        self.__build_gui()

    def __discard_images(self):
        images_names = {os.path.splitext(x)[0]: x for x in self.images}
        images_labeled = {os.path.splitext(x)[0] for x in self.already_labeled}
        images_ids = set(images_names.keys())
        remaining = images_ids - images_labeled
        self.images = list()
        for current in remaining:
            try:
                image_name = images_names[current]
                self.images.append(image_name)
            except Exception:
                pass

    def __build_gui(self):
        # Display image
        try:
            image = Image.open(join(self.root_directory, self.images[0]))
        except IndexError:
            mb.showwarning('Yes', 'No images found')
            sys.exit(-1)
        image.thumbnail(self.image_size)
        image = ImageTk.PhotoImage(image)
        self.label_image = Label(image=image)
        self.label_image.image = image
        self.label_image.grid(row=self.__apply_position(), column=1, columnspan=3)
        # Display group of labels
        for index in range(len(self.options_components)):
            text = self.options_components[str(index)]['text']
            options = self.options_components[str(index)]['options']
            action = self.option_actions[index]
            group = LabelFrame(self.root, text=text)
            group.grid(row=self.__apply_position(), column=0, columnspan=3)
            frame_counter = 1
            for option in options:
                Radiobutton(group, text=option, value=frame_counter-1,
                            variable=action).grid(row=frame_counter, column=0, columnspan=3)
                frame_counter += 1
        # Display control buttons
        Button(
            self.root, text=Components.BUTTON_BACK_TEXT,
            command=lambda: self.__show_image(self.images[self.__decrement_counter()])
        ).grid(row=self.component_gui_position, column=0)
        Button(
            self.root, text=Components.BUTTON_FORWARD_TEXT,
            command=lambda: self.__show_image(self.images[self.__increment_counter()])
        ).grid(row=self.component_gui_position, column=1)
        Button(
            self.root, text=Components.BUTTON_LABEL_TEXT,
            command=lambda: self.__label_image(self.images[self.image_index])
        ).grid(row=self.component_gui_position, column=2)

    def __apply_position(self):
        self.component_gui_position += 1
        return self.component_gui_position - 1

    def init_app(self):
        self.root.mainloop()

    def __show_image(self, image_name):
        self.label_image.grid_forget()
        image = Image.open(join(self.root_directory, image_name))
        image.thumbnail(self.image_size)
        image = ImageTk.PhotoImage(image)
        self.label_image.configure(image=image)
        self.label_image.image = image
        self.label_image.grid(row=0, column=1, columnspan=3)

    def __increment_counter(self):
        max_images = len(self.images)
        if self.image_index >= max_images - 1:
            self.image_index = -1
        self.image_index += 1
        return self.image_index

    def __decrement_counter(self):
        max_images = len(self.images)
        self.image_index -= 1
        if self.image_index < 0:
            self.image_index = max_images - 1
        return self.image_index

    def __label_image(self, image_name):
        selections = [selection.get() for selection in self.option_actions]
        groups = [option['text'] for _, option in self.options_components.items()]
        update = {groups[i]: selections[i] for i in range(len(selections))}
        print(image_name, update)
        self.images_scores.update({image_name: update})
        name, _ = splitext(image_name)
        file = open('{}.json'.format(join(self.root_directory, name)), 'w', encoding='utf-8')
        file.write(json.dumps(self.images_scores[image_name], indent=4))
        file.close()
