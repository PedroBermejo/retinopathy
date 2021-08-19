import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.functional import cross_entropy
from torch.optim import Adam
from pytorch_lightning.core import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class NetModel(LightningModule):

    def __init__(self):
        super(NetModel, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 5)
        # self.conv2 = nn.Conv2d(32, 16, 5)
        # self.conv3 = nn.Conv2d(16, 16, 5)
        # self.pool1 = nn.MaxPool2d(2)
        # self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(1)
        # # self.sigm = nn.Sigmoid()
        # # self.softmax2 = nn.Softmax(1)
        # self.fc1 = nn.Linear(16 * 12 * 12, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 2)
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Linear(2048, 2)
        # self.model.fc = to

    def forward(self, x):
        # y = self.pool1(self.relu(self.conv1(x)))
        # y = self.pool1(self.relu(self.conv2(y)))
        # y = self.pool1(self.relu(self.conv3(y)))
        # y = y.view(-1, 12 * 12 * 16)  # Flatten
        # y = self.relu(self.fc1(y))
        # y = self.relu(self.fc2(y))
        # y = self.softmax(self.fc3(y))
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        image, target = batch
        y = self(image)
        loss = cross_entropy(y, target)
        acc = accuracy(y, target)
        self.log('acc', acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        avg_acc = torch.stack([output['acc'] for output in outputs]).mean()
        metrics = {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        image, target = batch
        y = self(image)
        loss = cross_entropy(y, target)
        acc = accuracy(y, target)
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        avg_acc = torch.stack([output['acc'] for output in outputs]).mean()
        metrics = {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(.5),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root='dataset/train', transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=4, num_workers=4)
        return loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root='dataset/validation', transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=4, num_workers=4)
        return loader
