import torch
import os
import cv2

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, transform):
        'Initialization'
        self.path = path
        self.transform = transform

        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

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