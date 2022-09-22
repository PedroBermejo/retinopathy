import torch
import os
import re
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch'
    def __init__(self, path, transform, label):
        'Initialization'
        self.path = path
        self.transform = transform
        self.label = label
        self.labels = {}
        self.imageNames = []
        
        images = [
            name for name in os.listdir(self.path)
            if not re.match(r'[\w,\d]+\.[json]{4}', name)
        ]

        for name in images:
            self.labels[name] = self.label

        self.imageNames = images
        # print(self.imageNames)
        # print(self.labels)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imageNames)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.imageNames[index]
        y = self.labels[ID]

        # Load data and get label
        img_name = os.path.join(self.path, ID)
            
        pillow_image = Image.open(img_name)
        img = np.array(pillow_image)
        image = self.transform(image=img)
        image = image['image']

        return image, y, ID