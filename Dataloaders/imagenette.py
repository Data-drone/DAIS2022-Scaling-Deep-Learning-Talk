from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision import transforms
import torchvision.datasets as datasets
import os

class ImagenetteDataModule(LightningDataModule):
    """
    
    A dataloader ckass for imagenette
    See: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    
    """
  
    name = 'imagenette'
    
    def __init__(self,
                data_dir: str,
                batch_size: int = 32,
                num_workers: int = 4,
                pin_memory: bool = False):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = 10
        self.image_shape = [3, 224, 224]

    def train_dataloader(self):
        train_dir = os.path.join(self.data_dir, 'train')

        train_dataset = datasets.ImageFolder(
          train_dir,
          transforms.Compose(
            [transforms.RandomResizedCrop(size=(224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), self.normalise]
          )
        )

        train_loader = DataLoader(
          dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=self.pin_memory
        )

        return train_loader
    
    def val_dataloader(self):
        val_dir = os.path.join(self.data_dir, 'val')

        val_dataset = datasets.ImageFolder(
          val_dir,
          transforms.Compose(
            [transforms.Resize(size=(224, 224)), transforms.ToTensor(), self.normalise]
          )
        )

        val_loader = DataLoader(
          dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=self.pin_memory
        )

        return val_loader
    