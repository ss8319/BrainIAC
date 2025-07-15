import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import mean_absolute_error
import wandb

# Import model and dataset classes from local files
from model import ViTBackboneNet, Classifier, SingleScanModel
from dataset import BrainAgeDataset , get_default_transform, get_validation_transform

class BrainAgeLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.backbone = ViTBackboneNet(
            simclr_ckpt_path=config['simclrvit']['ckpt_path']
        )
        self.classifier = Classifier(d_model=768, num_classes=1)  # 768 for ViT-B, 1 for regression
        self.model = SingleScanModel(self.backbone, self.classifier)
        self.criterion = nn.MSELoss()
        self.best_val_mae = float('inf')
        self.validation_step_outputs = []

        # Freeze backbone if specified
        if str(config['train']['freeze']).lower() == "yes":
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen!!")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        output = {'val_loss': loss.detach(), 'y_true': y.detach().cpu(), 'y_pred': y_hat.detach().cpu()}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        y_true = torch.cat([o['y_true'] for o in self.validation_step_outputs]).numpy().flatten()
        y_pred = torch.cat([o['y_pred'] for o in self.validation_step_outputs]).numpy().flatten()
        
        mae = mean_absolute_error(y_true, y_pred)
        self.log('val_mae', mae, prog_bar=True)
        
        if mae < self.best_val_mae:
            self.best_val_mae = mae
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Pass only the trainable parameters to the optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['optim']['lr'], weight_decay=self.config['optim']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

class BrainAgeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        image_size = tuple(self.config['data']['size'])
        self.train_dataset = BrainAgeDataset(
            csv_path=self.config['data']['csv_file'],
            root_dir=self.config['data']['root_dir'],
            transform=get_default_transform(image_size=image_size)
        )
        self.val_dataset = BrainAgeDataset(
            csv_path=self.config['data']['val_csv'],
            root_dir=self.config['data']['root_dir'],
            transform=get_validation_transform(image_size=image_size)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=self.config['data']['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=1)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config_finetune.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Wandb logger
    wandb_logger = WandbLogger(
        project=config['logger']['project_name'],
        name=config['logger']['run_name'],
        config=config
    )

    # Data and model
    data_module = BrainAgeDataModule(config)
    model = BrainAgeLightningModule(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logger']['save_dir'],
        filename=config['logger']['save_name'],
        monitor='val_mae',
        mode='min',
        save_top_k=5
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config['model']['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu',
        devices=1,  # Use only one GPU
        strategy='ddp_find_unused_parameters_true',  # Changed from 'ddp' to enable unused parameter detection
        precision="16-mixed"  # Enable mixed precision training
    )

    trainer.fit(model, datamodule=data_module) 