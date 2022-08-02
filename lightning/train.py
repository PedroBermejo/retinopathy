from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
import argparse
import importlib
import os
import time


def main():
    template_model = getattr(importlib.import_module(args.model), 'NetModel')
    start_time = time.time()
    model = template_model(
        os.path.join(os.getcwd(), args.train_path),
        os.path.join(os.getcwd(), args.val_path))
    print(model)
    board = TensorBoardLogger(save_dir=os.path.join(os.getcwd(), args.save_path))
    checkpoint = ModelCheckpoint(
        filename='model-' + args.model + '-{epoch:02d}', monitor='val_loss', verbose=True, mode='min'
    )
    trainer = Trainer(
        max_epochs=2, gpus=1, default_root_dir=os.path.join(os.getcwd(), args.save_path),
        logger=[board], callbacks=[checkpoint]
    )

    trainer.fit(model=model)

    print('Training took {} seconds'.format(int(time.time() - start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', help='path for images for training')
    parser.add_argument('--val-path', help='path for images for validation')
    parser.add_argument('--model', help='model name: inceptionV3, mobilenetV2, resnet50, vgg19')
    parser.add_argument('--save-path', help='Folder to save logs and checkpoints')
    args = parser.parse_args()
    main()
