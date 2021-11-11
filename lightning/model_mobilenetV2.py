import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.functional import cross_entropy
from torch.optim import Adam
from pytorch_lightning.core import LightningModule
import torchmetrics
from torch.utils.data import DataLoader
import albumentations
import albumentations.pytorch.transforms as AT
from images_dataset import Dataset


class NetModel(LightningModule):

    def __init__(self, train_path, val_path):
        super(NetModel, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.softmax = nn.LogSoftmax(1)
        self.model = models.mobilenet_v2(pretrained=True)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.softmax(self.model(x))

    def training_step(self, batch, batch_idx):
        image, target = batch
        y = self(image)
        loss = cross_entropy(y, target)
        acc = self.accuracy(y, target)
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
        acc = self.accuracy(y, target)
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
        transform = albumentations.Compose([
            albumentations.Resize(width=100, height=100),
            albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            AT.ToTensorV2()
        ])
        dataset = Dataset(path=self.train_path, transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=4, num_workers=4)
        return loader

    def val_dataloader(self):
        transform = albumentations.Compose([
            albumentations.Resize(width=100, height=100),
            albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            AT.ToTensorV2()
        ])
        dataset = Dataset(path=self.val_path, transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=4, num_workers=4)
        return loader