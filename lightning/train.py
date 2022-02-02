from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model_vgg19 import NetModel

import os

# Probar dataset 70 train 30 val y 80, 20
if __name__ == '__main__':
    scriptDir = os.path.dirname(__file__)
    train_path = os.path.join(scriptDir, '../../retinopathy-dataset/train/')
    val_path = os.path.join(scriptDir, '../../retinopathy-dataset/val/')

    model = NetModel(train_path, val_path)
    print(model)
    board = TensorBoardLogger(save_dir='lightning/board')
    checkpoint = ModelCheckpoint(
        filename='model-{epoch:02d}', monitor='val_loss', verbose=True, mode='min'
    )
    trainer = Trainer(
        max_epochs=5, default_root_dir='lightning/results', logger=[board], callbacks=[checkpoint]
    )
    trainer.fit(model=model)
