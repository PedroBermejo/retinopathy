import os
import torch
import numpy as np

from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from argparse import Namespace
from typing import Type
from scipy.ndimage import label
from PIL import Image
from cropmodule import CropRetinaModule
from torchvision.transforms.functional import to_tensor
from common import create_directory
from tqdm import tqdm


class Cropper:

    def __init__(self, model_path, src, dst) -> None:
        self.model_name = os.path.join(os.getcwd(), model_path)
        self.src = os.path.join(os.getcwd(), src)
        self.dst = os.path.join(os.getcwd(), dst)

    def __load_model(self) -> Module:
        model = None
        try:
            model = CropRetinaModule.load_from_checkpoint(self.model_name)
            # model.cuda()
            model.eval()
            model.freeze()
        except Exception as e:
            raise e
        return model

    def __load_image(self, imagename: str) -> ndarray:
        image = Image.open(imagename).convert('L')
        image = image.convert('RGB')
        image = image.resize((512, 512))
        image = np.array(image)
        return image

    def __convert_to_tensor(self, image: ndarray) -> Type[Tensor]:
        tensor = to_tensor(image)
        tensor = torch.unsqueeze(tensor, dim=0)
        tensor = tensor.float()
        # tensor = tensor.cuda()
        return tensor

    @classmethod
    def __normalize(cls, image: ndarray) -> ndarray:
        image = image.astype(np.float32) / 255
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32)
        return image

    @classmethod
    def __remove_remaining(cls, data_image: ndarray) -> ndarray:
        labels, _ = label(data_image)
        height, width = data_image.shape
        groups = np.unique(labels)
        zeros_counter = data_image[(data_image == 0)].size
        zeros_id = 0
        # Remove small groups of pixels
        remaining_groups = list()
        for group in groups:
            count = labels[(labels == group)].size
            if count == zeros_counter:
                zeros_id = group
            percentage = count / (height * width)
            if percentage < 0.30:
                remaining_groups.append(group)

        for group in remaining_groups:
            labels = np.where(labels == group, zeros_id, labels)
        # Reconvert to binary format
        labels = np.where(labels != zeros_id, 1, labels)
        labels = np.where(labels == zeros_id, 0, labels)
        labels = labels.astype(np.uint8)
        return labels

    @classmethod
    def __add_zero_banners(cls, data_image: ndarray) -> ndarray:
        axis = 0
        height, width, channels = data_image.shape
        b = abs(int(height - width))
        b1 = int(b / 2)
        b2 = b - b1
        if height > width:
            axis = 1
            z1 = np.zeros((height, b1, channels))
            z2 = np.zeros((height, b2, channels))
        elif width > height:
            z1 = np.zeros((b1, width, channels))
            z2 = np.zeros((b2, width, channels))
        else:
            return data_image
        new_img = np.append(data_image, z1, axis=axis)
        new_img = np.append(z2, new_img, axis=axis)
        new_img = new_img.astype(np.uint8)
        return new_img

    @classmethod
    def is_retina_mask_empty(cls, retina_mask: ndarray) -> bool:
        for mask in retina_mask:
            if mask.size == 0:
                return True
        return False

    def __predict(self, model: Type[Module], tensor: Type[Tensor]):
        output = model(tensor)
        output = torch.squeeze(output)
        output = output.detach().cpu().numpy()
        output = np.where(output >= 0.5, 1., 0.)
        return output

    def crop(self) -> None:
        model = self.__load_model()
        assert model is not None, 'Was not able to load model!'
        create_directory(self.dst)
        for root, _, files in os.walk(self.src):
            for file_name in tqdm(sorted(files)):
                image = self.__load_image(imagename=os.path.join(root, file_name))
                original_image = Image.open(os.path.join(root, file_name))
                height, width = original_image.size
                normalized = self.__normalize(image=image)
                tensor = self.__convert_to_tensor(image=normalized)
                prediction = self.__predict(model=model, tensor=tensor)
                clean_mask = self.__remove_remaining(data_image=prediction)
                mask = Image.fromarray(clean_mask)
                mask = mask.resize((height, width))
                mask = np.array(mask)
                pos = np.where(mask)
                if self.is_retina_mask_empty(pos):
                    continue
                x_min = np.min(pos[1])
                x_max = np.max(pos[1])
                y_min = np.min(pos[0])
                y_max = np.max(pos[0])
                original_image = np.array(original_image)
                mask = np.stack([mask]*3, axis=-1)
                original_image = original_image * mask
                cropped = original_image[y_min:y_max, x_min:x_max]
                cropped = self.__add_zero_banners(data_image=cropped)
                cropped = Image.fromarray(cropped)
                cropped = cropped.resize((512, 512))
                name, _ = os.path.splitext(file_name)
                cropped.save(os.path.join(self.dst, name + '.png'), 'PNG')