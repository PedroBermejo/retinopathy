from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import NetModel


if __name__ == '__main__':
    model = NetModel()
    print(model)
    board = TensorBoardLogger(save_dir='board')
    checkpoint = ModelCheckpoint(
        filename='model-{epoch:02d}', monitor='val_loss', verbose=True, mode='min'
    )
    trainer = Trainer(
        max_epochs=10, gpus=1, default_root_dir='results', logger=[board], callbacks=[checkpoint]
    )
    trainer.fit(model=model)
