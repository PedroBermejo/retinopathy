import torch
import torchvision.transforms.functional as functional

from PIL import Image
from model import NetModel
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


if __name__ == '__main__':
    model = NetModel()
    new_model = model.load_from_checkpoint(checkpoint_path='results/default/3/checkpoints/model-epoch=00.ckpt')

    # image_name = 'dataset/validation/dogs/dog.1002.jpg'
    # image = Image.open(image_name)
    # image = image.resize((128, 128))
    # tensor = functional.to_tensor(image)
    # tensor = torch.unsqueeze(tensor, dim=0)
    #
    # y = new_model(tensor)
    # out = torch.argmax(y, 1)
    # print(y)
    # print(out)

    transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root='dataset/validation', transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=1)

    for images, targets in loader:
        y = new_model(images)
        out = torch.argmax(y, 1)
        npout = out.cpu().detach().numpy()
        nptar = targets.cpu().detach().numpy()
        print('target: {} - predict: {}'.format(nptar, npout))

    