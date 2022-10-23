import os
import numpy as np

from PIL import Image
from os.path import join
from torch.utils.data import Dataset


class CropRetinaDataset(Dataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images = [join(images_dir, image_id) for image_id in self.ids]
        self.masks = [join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [1]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('L')
        image = image.convert('RGB')
        image = image.resize((512, 512))
        image = np.array(image)
        mask = Image.open(self.masks[idx])
        mask = mask.resize((512, 512))
        mask = np.array(mask)
        mask = np.where(mask != 0, 1, 0)
        mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype(np.float32)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
