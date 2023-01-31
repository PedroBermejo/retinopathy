import torch

from images_dataset_predict import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch.transforms as AT
import argparse
import importlib
import os
import pandas as pd


def get_dataset_loaders():
    transform = A.Compose([
        A.Resize(width=300, height=300),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        AT.ToTensorV2()
    ])

    good_dataset = Dataset(os.path.join(os.getcwd(), args.good_path), transform, 1)
    blur_dataset = Dataset(os.path.join(os.getcwd(), args.blur_path), transform, 0)
    gauss_dataset = Dataset(os.path.join(os.getcwd(), args.gauss_path), transform, 0)
    fog_dataset = Dataset(os.path.join(os.getcwd(), args.fog_path), transform, 0)

    good_loader = DataLoader(dataset=good_dataset, shuffle=False, batch_size=100, num_workers=4)
    blur_loader = DataLoader(dataset=blur_dataset, shuffle=False, batch_size=100, num_workers=4)
    gauss_loader = DataLoader(dataset=gauss_dataset, shuffle=False, batch_size=100, num_workers=4)
    fog_loader = DataLoader(dataset=fog_dataset, shuffle=False, batch_size=100, num_workers=4)

    loaders = {
        'good_images': good_loader,
        'blur': blur_loader,
        'gauss_noise': gauss_loader,
        'random_fog': fog_loader
    }
    return loaders


def inception_model():
    inception_template = getattr(importlib.import_module('inceptionV3'), 'NetModel')
    return inception_template.load_from_checkpoint(os.path.join(os.getcwd(), args.inceptionV3_model_path),
                                                              train_path='', val_path='')


def mobilenet_model():
    mobilenet_template = getattr(importlib.import_module('mobilenetV2'), 'NetModel')
    return mobilenet_template.load_from_checkpoint(os.path.join(os.getcwd(), args.mobilenetV2_model_path),
                                                          train_path='', val_path='')


def resnet_model():
    resnet_template = getattr(importlib.import_module('resnet50'), 'NetModel')
    return resnet_template.load_from_checkpoint(os.path.join(os.getcwd(), args.resnet50_model_path),
                                                        train_path='', val_path='')


def vgg_model():
    vgg_template = getattr(importlib.import_module('vgg19'), 'NetModel')
    return vgg_template.load_from_checkpoint(os.path.join(os.getcwd(), args.vgg19_model_path),
                                                  train_path='', val_path='')



def main():
    models = {
        'inception': inception_model,
        'mobilenet': mobilenet_model,
        'resnet': resnet_model,
        'vgg': vgg_model
    }

    # for name, loader in loaders.items():
    #     images, labels, image_names = next(iter(loader))
    #     print(name)
    #     print("Length: ", len(images))
    #     print("Label: ", labels[0])
    #     print("Image_name: ", image_names[0])
    #     print()

    for model_name, model_function in models.items():
        df = pd.DataFrame()
        model = model_function()
        loaders = get_dataset_loaders()
        print(model_name)
        for loader_name, loader in loaders.items():
            print(loader_name)
            for images, labels, image_names in iter(loader):
                probability = model(images)
                torch.exp(probability)
                predicts = [0 if value[0] > value[1] else 1 for value in probability]
                # print(predicts)
                temp = pd.DataFrame()
                temp[f'{loader_name}_{model_name}_names'] = image_names
                temp[f'{loader_name}_{model_name}_probability_left'] = probability.detach().numpy()[:, 0]
                temp[f'{loader_name}_{model_name}_probability_right'] = probability.detach().numpy()[:, 1]
                temp[f'{loader_name}_{model_name}_predicts'] = predicts
                frames = [df, temp]
                df = pd.concat(frames, ignore_index=True)
                print(len(df.index))

        df.to_csv(os.path.join(os.getcwd(), args.path_to_csv + model_name + '.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--good-path', help='path for images for training')
    parser.add_argument('--blur-path', help='path for images for training')
    parser.add_argument('--gauss-path', help='path for images for training')
    parser.add_argument('--fog-path', help='path for images for training')
    parser.add_argument('--inceptionV3-model-path', help='Path where checkpoint (model) is saved')
    parser.add_argument('--mobilenetV2-model-path', help='Path where checkpoint (model) is saved')
    parser.add_argument('--resnet50-model-path', help='Path where checkpoint (model) is saved')
    parser.add_argument('--vgg19-model-path', help='Path where checkpoint (model) is saved')
    parser.add_argument('--path-to-csv', help='Path where to save csv')

    args = parser.parse_args()
    main()
