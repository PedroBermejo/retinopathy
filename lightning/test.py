import torch
import os
import torchvision.transforms.functional as functional

from PIL import Image
from model_inceptionV2 import NetModel
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


if __name__ == '__main__':
    model = NetModel('', '')
    #new_model = model.load_from_checkpoint(checkpoint_path='results/default/3/checkpoints/model-epoch=00.ckpt')

    scriptDir = os.path.dirname(__file__)
    image_name = 'test-data/10_left.jpeg'
    image = Image.open(os.path.join(scriptDir, image_name))
    image = image.resize((300, 300))
    tensor = functional.to_tensor(image)
    tensor = torch.unsqueeze(tensor, dim=0)
    print(tensor.shape)

    y = model(tensor)
    out = torch.argmax(y, 1)
    print(y.shape)
    print("y", y)
    print(out.shape)
    print("out", out)

'''
    transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root='dataset/validation', transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=1)

    for images, targets in loader:
        y = model(images)
        out = torch.argmax(y, 1)
        npout = out.cpu().detach().numpy()
        nptar = targets.cpu().detach().numpy()
        print('target: {} - predict: {}'.format(nptar, npout))

'''  