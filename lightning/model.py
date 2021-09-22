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

    def __init__(self, train_path, val_path):
        super(NetModel, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        # Jugar con los kernels
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 5)
        self.conv3 = nn.Conv2d(32, 16, 5)
        self.conv4 = nn.Conv2d(16, 8, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(1)
        #self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        # Reducir una fc capa, son pesadas
        self.fc1 = nn.Linear(8 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        # Probar Image Net (200 size?), Inception_v4, ResNet_, MobileNet, EficientNet
        #self.model = models.inception_v3(pretrained=True, aux_logits=False)
        #self.model.fc = nn.Linear(2048, 2)
        # self.model.fc = to

    def forward(self, x):
        # Jugar con el pooling
        y = self.pool(self.relu(self.conv1(x)))
        y = self.pool(self.relu(self.conv2(y)))
        y = self.pool(self.relu(self.conv3(y)))
        y = self.pool(self.relu(self.conv4(y)))
        #print("Before flatten", y.shape)
        y = y.view(-1, 8 * 28 * 28)  # Flatten
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.relu(self.fc4(y))
        y = self.softmax(self.fc5(y))
        # y = self.model(x)
        return y

    # ver como vizualizar o agrupar por epoca y no por paso
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

    # Jugar con el LR
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    # Probar diferentes transformaciones, cambio de brillo, iluminacion, contraste, desenfoque(blur)
    # normalizaciones(imagenet), aumentar resolucion imagen 
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize([512, 512]),
            # ver porcentaje
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(root=self.train_path, transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=4, num_workers=2)
        return loader


    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(root=self.val_path, transform=transform)
        loader = DataLoader(dataset=dataset, shuffle=False, batch_size=4, num_workers=2)
        return loader
