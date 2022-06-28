from petastorm import TransformSpec
from PIL import Image
from torchvision import transforms
import numpy as np
import io
import pytorch_lightning as pl

from pyspark.sql.functions import col, pandas_udf, PandasUDFType


class ImagenettePetaDataModule(pl.LightningDataModule):
  
  def __init__(self, train_converter, val_converter, device_id:int=0, device_count:int=1, batch_size=32):
    
    self.train_converter = train_converter
    self.val_converter = val_converter
    self.train_dataloader_context = None
    self.val_dataloader_context = None
    self.prepare_data_per_node = False
    self._log_hyperparams = False
    self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    self.device_id = device_id
    self.device_count = device_count
    self.image_shape = {3, 224, 224}
    self.batch_size = batch_size
  
    
  def train_dataloader(self):
    if self.train_dataloader_context:
        self.train_dataloader_context.__exit__(None, None, None)
    self.train_dataloader_context = self.train_converter.make_torch_dataloader(transform_spec=self._get_transform_train_spec(), 
                                                                               num_epochs=None,
                                                                               cur_shard=self.device_id, 
                                                                               shard_count=self.device_count, 
                                                                               batch_size=self.batch_size*self.device_count)
    return self.train_dataloader_context.__enter__()
  
  def val_dataloader(self):
    if self.val_dataloader_context:
        self.val_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context = self.val_converter.make_torch_dataloader(transform_spec=self._get_transform_val_spec(), 
                                                                           num_epochs=None, 
                                                                           cur_shard=self.device_id, 
                                                                           shard_count=self.device_count,
                                                                           batch_size=self.batch_size*self.device_count)
    return self.val_dataloader_context.__enter__()
    
  def teardown(self, stage=None):
    # Close all readers (especially important for distributed training to prevent errors)
    self.train_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context.__exit__(None, None, None)
  
  def preprocess_train(self, img):
    
    image = Image.open(io.BytesIO(img))
    transform = transforms.Compose([
      transforms.RandomResizedCrop(size=(224, 224)), 
      transforms.RandomHorizontalFlip(), 
      transforms.ToTensor(), 
      self.normalise
    ])
    
    return transform(image)
  
  def preprocess_val(self, img):
    
    image = Image.open(io.BytesIO(img))
    transform = transforms.Compose([
      transforms.Resize(size=(224, 224)), 
      transforms.ToTensor(), 
      self.normalise
    ])
    
    return transform(image)
  
  def _transform_train_rows(self, batch):
    
    # To keep things simple, use the same transformation both for training and validation
    batch["features"] = batch["content"].map(lambda x: self.preprocess_train(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    
    return batch
  
  def _transform_val_rows(self, batch):
    
    # To keep things simple, use the same transformation both for training and validation
    batch["features"] = batch["content"].map(lambda x: self.preprocess_val(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    return batch

  def _get_transform_val_spec(self):
    return TransformSpec(self._transform_val_rows, 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "label"])
  
  def _get_transform_train_spec(self):
    return TransformSpec(self._transform_train_rows, 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "label"])