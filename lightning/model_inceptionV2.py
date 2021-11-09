import torch
import torch.nn as nn
import torchvision.models as models

from albumentations.core.composition import OneOf
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from pytorch_lightning.core import LightningModule
#from pytorch_lightning.metrics.functional import accuracy
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class NetModel(LightningModule):

    def __init__(self, train_path, val_path):
        super(NetModel, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.softmax = nn.LogSoftmax(1)
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Linear(2048, 2)


    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        image, target = batch
        y = self(image)
        loss = cross_entropy(y, target)
        acc = torchmetrics.Accuracy()(y, target)
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
        acc = torchmetrics.Accuracy()(y, target)
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
        dataset = ImageFolder(root=self.train_path, transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=4, num_workers=2)
        return loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root=self.val_path, transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=4, num_workers=2)
        return loader
