# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Scaling on Single Node
# MAGIC 
# MAGIC As we discussed, it is more efficient to scale on single node first

# COMMAND ----------

# Loading Libs
from sparkdl import HorovodRunner
import horovod.torch as hvd

from TrainLoop.pl_train import main_hvd, build_trainer, main_train
import os

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Configuration Parameters

# COMMAND ----------

# MLFlow workaround
#DEMO_SCOPE_TOKEN_NAME = "TOKEN"
databricks_host = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")
databricks_token = dbutils.secrets.get(scope="scaling_dl", key="host_token")
os.environ['DATABRICKS_HOST'] = databricks_host
os.environ['DATABRICKS_TOKEN'] = databricks_token

imagenette_data_path = '/dbfs/Users/brian.law@databricks.com/data/imagenette2'

dais_root_folder = '/dbfs/Users/brian.law@databricks.com/dais_exp'

# We set the id so that we can have multiple notebooks all feeding in 
dais_experiment_id = 3156978719066434

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load Modules

# COMMAND ----------

from Models.resnet import ResnetClassification
from Dataloaders.imagenette import ImagenetteDataModule

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Experiments

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Baseline

# COMMAND ----------

batch_size = 32
epochs=10

# COMMAND ----------

data_module = ImagenetteDataModule(imagenette_data_path)
model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(data_module, model, 
           num_gpus=1, root_dir=dais_root_folder, 
           epoch=3, strat=None, 
           experiment_id=dais_experiment_id, 
           run_name='baseline_run')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Optimising Default Loader
# MAGIC 
# MAGIC We can optimise even the default dataloader to work faster. The keys to this are two variables:
# MAGIC The num_workers which is the CPU cores that the dataloader will utilise
# MAGIC And also `pin_memory` which helps to optimise the data transfer between cpu and gpu.
# MAGIC 
# MAGIC We can also look to increase the batch size to fill up our GPU ram

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(data_module, model, 
           num_gpus=1, root_dir=dais_root_folder, 
           epoch=3, strat=None, 
           experiment_id=dais_experiment_id, 
           run_name='baseline_single_tune')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Leverage PyTorch Lightning AutoTune
# MAGIC 
# MAGIC PyTorch Lighting has an autotune capability that can help us tofind the best batch size as well as the best learning rate
# MAGIC This will become more important later

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64
run_name = 'baseline_pl_autotune'

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

trainer = build_trainer(
  num_gpus=1, 
  root_dir=dais_root_folder, 
  epoch=3,
  strat=None,
  run_name=run_name,
  auto_scale_batch_size=None,
  auto_lr_find=True,
  precision=16 # 16 bit precision is faster
)

# COMMAND ----------

trainer.tune(model, datamodule=data_module)

# COMMAND ----------

# Since we split out the Trainer object we now need to manually fit
import mlflow

with mlflow.start_run(experiment_id=dais_experiment_id, run_name=run_name) as run:
            
  mlflow.log_param("model", model.model_tag)

  trainer.fit(model, data_module)

  # log model
  mlflow.pytorch.log_model(model, "models")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Adding GPUs

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64
num_gpus = 2
strat = 'dp'
run_name = 'dual_gpu_pl_autotune'

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

trainer = build_trainer(
  num_gpus=num_gpus, 
  root_dir=dais_root_folder, 
  epoch=3,
  strat=strat,
  run_name=run_name,
  auto_scale_batch_size=None,
  auto_lr_find=True,
  precision=16 # 16 bit precision is faster
)

trainer.tune(model, datamodule=data_module)

# COMMAND ----------

# Since we split out the Trainer object we now need to manually fit
import mlflow

with mlflow.start_run(experiment_id=dais_experiment_id, run_name=run_name) as run:
            
  mlflow.log_param("model", model.model_tag)

  trainer.fit(model, data_module)

  # log model
  mlflow.pytorch.log_model(model, "models")

# COMMAND ----------


