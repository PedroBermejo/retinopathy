from pytorch_lightning.core import LightningModule

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        return x
    
    def configure_optimizers(self):
        pass
       
    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, valid_batch, batch_idx):
        pass