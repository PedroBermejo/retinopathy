import torch
import os
import cv2
import re

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
            if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
        ]

        listBadImages = [
            os.path.splitext(name)[0] for name in os.listdir(os.path.join(self.path, 'bad'))
            if re.match(r'[\w,\d]+\.[json|JSON]{4}', name)
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
        ID = self.list_IDs[index]

        # Load data and get label
        img_name = os.path.join(self.path, ID)
        img = cv2.imread(img_name)
        image = self.transform(image=img)
        image = image['image']

        y = self.labels[ID]

        return image, y