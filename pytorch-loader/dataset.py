import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from os.path import join
from os.path import splitext
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class MyDataset(Dataset):
    ALLOWED_FORMATS = ['.jpeg', '.jpg']

    def __init__(self, image_path, tranform=None):
        self.image_path = image_path
        self.tranform = tranform
        self.image_names = list()
        for root, _, files in os.walk(image_path):
            for file_name in files:
                _, ext = splitext(file_name)
                if ext not in self.ALLOWED_FORMATS:
                    continue
                full_path = join(root, file_name)
                self.image_names.append(full_path)

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        image = Image.open(file_name)
        image = np.array(image)
        label = int(file_name.split('/')[-2])
        if self.tranform:
            image = self.tranform(image=image)['image']
        label = torch.as_tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.image_names)
