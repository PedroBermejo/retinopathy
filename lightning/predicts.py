from images_dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch.transforms as AT
import argparse
import importlib
import os
import pandas as pd


def main():
    inception_template = getattr(importlib.import_module('inceptionV3'), 'NetModel')
    mobilenet_template = getattr(importlib.import_module('mobilenetV2'), 'NetModel')
    resnet_template = getattr(importlib.import_module('resnet50'), 'NetModel')
    vgg_template = getattr(importlib.import_module('vgg19'), 'NetModel')

    inception_model = inception_template.load_from_checkpoint(os.path.join(os.getcwd(), args.inceptionV3_model_path),
                                                              train_path='', val_path='')
    mobilenet_model = mobilenet_template.load_from_checkpoint(os.path.join(os.getcwd(), args.mobilenetV2_model_path),
                                                              train_path='', val_path='')
    resnet_model = resnet_template.load_from_checkpoint(os.path.join(os.getcwd(), args.resnet50_model_path),
                                                        train_path='', val_path='')
    vgg_model = vgg_template.load_from_checkpoint(os.path.join(os.getcwd(), args.vgg19_model_path),
                                                  train_path='', val_path='')

    # print(inception_model)
    # print(mobilenet_model)
    # print(resnet_model)
    # print(vgg_model)

    transform = A.Compose([
        A.Resize(width=300, height=300),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        AT.ToTensorV2()
    ])

    good_dataset = Dataset(path=os.path.join(os.getcwd(), args.good_path), transform=transform)
    blur_dataset = Dataset(path=os.path.join(os.getcwd(), args.blur_path), transform=transform)
    gauss_dataset = Dataset(path=os.path.join(os.getcwd(), args.gauss_path), transform=transform)
    fog_dataset = Dataset(path=os.path.join(os.getcwd(), args.fog_path), transform=transform)

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

    for name, loader in loaders:
        images, labels = next(iter(loader))
        print(name)
        print(len(images))
        print(images[0])
        print(len(labels))
        print(labels[0])
        print()

    '''

    predicted = []
    all_labels = []
    for images, labels in iter(loader):
        predict = model(images)
        predict = [0 if value[0] > value[1] else 1 for value in predict]
        predicted = predicted + predict
        all_labels = all_labels + labels.detach().numpy().tolist()

    print(len(predicted))
    print(len(all_labels))

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    print("Print confussion matrix")
    print(confusion_matrix(all_labels, predicted))

    print("Print classification report")
    print(classification_report(all_labels, predicted))
    '''


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

    args = parser.parse_args()
    main()

