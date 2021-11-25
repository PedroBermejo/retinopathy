import torch
import os
import re
from PIL import Image
import numpy as np

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, transform):
        'Initialization'
        self.path = path
        self.transform = transform
        self.labels = {}
        self.imageNames = []
        
        listGoodImages = [
            os.path.splitext(name)[0] for name in os.listdir(os.path.join(self.path, 'good'))
            if re.match(r'[\w,\d]+\.[json|JSON|jpeg]{4}', name)
        ]

        listBadImages = [
            os.path.splitext(name)[0] for name in os.listdir(os.path.join(self.path, 'bad'))
            if re.match(r'[\w,\d]+\.[json|JSON|jpeg]{4}', name)
        ]

        for name in listGoodImages: 
            self.labels[name] = 1

        for name in listBadImages: 
            self.labels[name] = 0

        self.imageNames = listGoodImages + listBadImages
        #print(self.imageNames)
        #print(self.labels)

        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.imageNames)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.imageNames[index]
        y = self.labels[ID]
        img_name = ''

        # Load data and get label
        if y == 1:
            img_name = os.path.join(self.path, 'good', ID + '.jpeg')
        else:
            img_name = os.path.join(self.path, 'bad', ID + '.jpeg')
            
        pillow_image = Image.open(img_name)
        img = np.array(pillow_image)
        image = self.transform(image=img)
        image = image['image']

        return image, y