import argparse
import torch
import segmentation_models_pytorch as smp

from os.path import join
from torch.utils.data import DataLoader
from retinasegmenterdataset import CropRetinaDataset
from augmentationcrop import get_training_augmentation, get_validation_augmentation, get_test_augmentation
from common import get_preprocessing
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from pytorch_lightning.core import LightningModule


class CropRetinaModule(LightningModule):
    ENCODER_NAME = 'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    def __init__(self, hparams):
        super(CropRetinaModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = smp.Unet(
            encoder_name=self.ENCODER_NAME,
            encoder_weights=self.ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER_NAME, self.ENCODER_WEIGHTS)
        self.loss_train = DiceLoss()
        self.iou_train = IoU(threshold=0.5)
        self.loss_val = DiceLoss()
        self.iou_val = IoU(threshold=0.5)
        self.loss_test = DiceLoss()
        self.iou_test = IoU(threshold=0.5)

    def forward(self, x):
        y = self.model(x)
        return y

    def configure_optimizers(self):
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = Adam(params, lr=self.hparams.learning_rate, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.loss_train(output, target)
        iou = self.iou_train(output, target)
        metrics = {
            'train_loss': loss,
            'train_iou': iou
        }
        self.log_dict(metrics, prog_bar=True)
        return {'loss': loss, 'train_loss': loss.detach(), 'train_iou': iou}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([output['train_loss'] for output in outputs]).mean()
        avg_iou = torch.stack([output['train_iou'] for output in outputs]).mean()
        metrics = {
            'train_loss_epoch': avg_loss,
            'train_iou_epoch': avg_iou
        }
        self.log_dict(metrics)

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.loss_val(output, target)
        iou = self.iou_val(output, target)
        metrics = {
            'val_loss': loss,
            'val_iou': iou
        }
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        avg_iou = torch.stack([output['val_iou'] for output in outputs]).mean()
        metrics = {
            'val_loss_epoch': avg_loss,
            'val_iou_epoch': avg_iou
        }
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.loss_test(output, target)
        iou = self.iou_test(output, target)
        metrics = {
            'test_loss': loss,
            'test_iou': iou
        }
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        avg_iou = torch.stack([output['test_iou'] for output in outputs]).mean()
        metrics = {
            'test_loss_epoch': avg_loss,
            'test_iou_epoch': avg_iou
        }
        self.log_dict(metrics)

    def train_dataloader(self):
        dataset = CropRetinaDataset(
            images_dir=join(self.hparams.data_path, 'train', 'images'),
            masks_dir=join(self.hparams.data_path, 'train', 'masks'),
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn)
        )
        loader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers
        )
        return loader

    def val_dataloader(self):
        dataset = CropRetinaDataset(
            images_dir=join(self.hparams.data_path, 'validation', 'images'),
            masks_dir=join(self.hparams.data_path, 'validation', 'masks'),
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn)
        )
        loader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers
        )
        return loader

    def test_dataloader(self):
        dataset = CropRetinaDataset(
            images_dir=join(self.hparams.data_path, 'test', 'images'),
            masks_dir=join(self.hparams.data_path, 'test', 'masks'),
            augmentation=get_test_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn)
        )
        loader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers
        )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, required=True)
        parser.add_argument('--metrics_path', type=str, required=True)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--num_workers', type=int, default=4)
        return parser
