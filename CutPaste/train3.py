import torch
from torchvision import transforms
import torch.utils.data
import pytorch_lightning as pl
from model import CutPasteNet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
import argparse
from BurgerDataTrain import BurgerData as BGTrain
from BurgerDataTest2 import BurgerData as BGTest
from pathlib import Path

# Parse command-line arguments for hyperparameters and configurations.
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.00003)
    parser.add_argument('--num_epochs', default=15)
    parser.add_argument('--num_gpus', default=2)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--log_dir', default=r'tb_logs')
    parser.add_argument('--log_dir_name', default=r'cutAndPaste')
    parser.add_argument('--checkpoint_filename', default=r'weights')
    parser.add_argument('--monitor_checkpoint', default=r'train_loss')
    parser.add_argument('--monitor_checkpoint_mode', default=r'min')
    args = parser.parse_args()
    return args

# Define a PyTorch Lightning Module for the CutPaste model.
class CutPaste(pl.LightningModule):
    def __init__(self, hparams):
        super(CutPaste, self).__init__()
        # Save hyperparameters for later reference.
        self.save_hyperparameters(hparams)
        # Initialize the model architecture.
        self.model = CutPasteNet()
        # Set the loss function.
        self.criterion = torch.nn.CrossEntropyLoss()

    # Define the training data loader.
    def train_dataloader(self):
        
        #white
        mean=[0.5815, 0.5940, 0.5015]
        std=[0.2716, 0.2812, 0.2710]
        
        #white_with_edges
        #mean=[0.6384, 0.6557, 0.5500]
        #std=[0.2846, 0.2897, 0.2772]

        # Create training dataset and DataLoader.
        dataset = BGTrain(
            imgSize=self.hparams.input_size,
            stride=112,
            image_folder='/home/shn/data/white/train',
            #image_folder='/home/shn/data/white_with_edges/train',
            transform=transforms.Normalize(mean=mean, std=std)
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=16,
            shuffle=True
        )
        return loader
    
    # Define the test data loader.
    def test_dataloader(self):
        #white
        mean=[0.5815, 0.5940, 0.5015]
        std=[0.2716, 0.2812, 0.2710]
        
        #white_with_edges
        #mean=[0.6384, 0.6557, 0.5500]
        #std=[0.2846, 0.2897, 0.2772]
        
        # Create testing dataset and DataLoader.
        dataset = BGTest(
            imgSize=self.hparams.input_size,
            stride=112,
            image_folder='/home/shn/white/ground_truth',
            #image_folder='/home/shn/white_with_edges/ground_truth',
            transform=transforms.Normalize(mean=mean, std=std)
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader
    
    # Forward pass through the model.
    def forward(self, x):
        logits, embeds = self.model(x)
        return logits, embeds 

    # Configure optimizer and learning rate scheduler.
    def configure_optimizers(self):
        #optimizer=optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999) ,
                             #weight_decay=self.hparams.weight_decay)
        # Use SGD optimizer with momentum and weight decay as specified in the hyperparameters
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                              momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
                
        # Use a cosine annealing scheduler with warm restarts for learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs)
        return [optimizer], [scheduler]

    # Callback at the start of training.
    def on_train_start(self):
        print('Starting training')

    # Callback at the start of testing.
    def on_test_start(self):
        print('Starting testing')

    # Define one training step.
    def training_step(self, batch, batch_idx):
        x = torch.cat(batch, axis=0)
        y = torch.arange(len(batch))
        y = y.repeat_interleave(len(batch[0])).cuda()
        logits, embeds = self(x)
        loss = self.criterion(logits, y)
        predicted = torch.argmax(logits, axis=1)
        accuracy = torch.true_divide(torch.sum(predicted == y), predicted.size(0))
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)
        
        return loss

    # Define one testing step.
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, embeds = self(x)
        loss = self.criterion(logits, y)

    # Callback at the end of training.
    def on_train_end(self):
        print("Training is ending")

    # Callback at the end of testing.
    def on_test_end(self):
        print("Testing is ending")

# Main execution: parse arguments, setup logging and checkpointing, and start training.
if __name__ == "__main__":
    args = get_args()
    # Set NAME_CKPT without args.dataset_path
    NAME_CKPT = args.checkpoint_filename
    # Initialize TensorBoard logger for training visualization.
    logger = TensorBoardLogger(args.log_dir, name=args.log_dir_name)
    checkpoint_dir = (
        Path(logger.save_dir)
        / logger.name
        / f"version_{logger.version}"
        / "checkpoints"
    )
    # Setup checkpointing based on monitored training loss.
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor_checkpoint,
        dirpath=str(checkpoint_dir),
        filename=NAME_CKPT, #Weights
        mode=args.monitor_checkpoint_mode #train_loss
    )
    # Instantiate the model with hyperparameters.
    model = CutPaste(hparams=args)
    num_devices = int(args.num_gpus)
    # Initialize the PyTorch Lightning trainer.
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu' if num_devices > 0 else 'cpu',
        devices=num_devices if num_devices > 0 else None,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epochs,
        enable_checkpointing=True,
    )
    # Start model training.
    trainer.fit(model)
