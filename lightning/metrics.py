from images_dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch.transforms as AT
import argparse
import importlib
import os
import time


def main():
    template_model = getattr(importlib.import_module(args.model), 'NetModel')
    model = template_model.load_from_checkpoint(os.path.join(os.getcwd(), args.model_path), train_path='', val_path='')
    print(model)

    transform = A.Compose([
        A.Resize(width=300, height=300),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        AT.ToTensorV2()
    ])

    dataset = Dataset(path=os.path.join(os.getcwd(), args.test_path), transform=transform)
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=100, num_workers=4)

    images, labels = next(iter(loader))
    print(len(images))
    print(images[0])
    print(len(labels))
    print(labels[0])

    start_time = time.time()

    predicted = []
    all_labels = []
    for images, labels in iter(loader):
        predict = model(images)
        predict = [ 0 if value[0] > value[1] else 1 for value in predict ]
        predicted = predicted + predict
        all_labels = all_labels + labels.detach().numpy().tolist()

    print(len(predicted))
    print(len(all_labels))
    print('Predict took {} seconds'.format(int(time.time() - start_time)))

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    print("Print confussion matrix")
    print(confusion_matrix(all_labels, predicted))

    print("Print classification report")
    print(classification_report(all_labels, predicted))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name: inceptionV3, mobilenetV2, resnet50, vgg19')
    parser.add_argument('--test-path', help='path for images for training')
    parser.add_argument('--model-path', help='Path where checkpoint (model) is saved')

    args = parser.parse_args()
    main()

