# Databricks notebook source
# MAGIC %pip install mosaicml

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Extra tests

# COMMAND ----------

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Parameters

# COMMAND ----------

num_workers = 8
batch_size = 128
imagenette_data_path = '/dbfs/Users/brian.law@databricks.com/data/imagenette2'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Define DataLoaders
# MAGIC 
# MAGIC We can still reuse the PyTorch Lightning one but we need to extract out the dataloaders

# COMMAND ----------

from Dataloaders.imagenette import ImagenetteDataModule

data_module = ImagenetteDataModule(imagenette_data_path, 
                                   num_workers=num_workers,
                                   batch_size=batch_size)

# COMMAND ----------

train_dl = data_module.train_dataloader()
val_dl = data_module.val_dataloader()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup Model

# COMMAND ----------

from composer.models import ComposerResNet
from composer import Trainer
from composer.algorithms import BlurPool, ChannelsLast, CutMix, LabelSmoothing


# COMMAND ----------

model = ComposerResNet(
  model_name = 'resnet50',
  num_classes = 10,
  pretrained = False
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Trainer

# COMMAND ----------

trainer = Trainer(
  model = model,
  train_dataloader = train_dl,
  eval_dataloader = val_dl, 
  max_duration = "15ep",
  algorithms = [
    BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True),
    ChannelsLast(),
    CutMix(num_classes=10),
    LabelSmoothing(smoothing=0.1)
  ]
)

# COMMAND ----------

trainer.fit()

# COMMAND ----------


