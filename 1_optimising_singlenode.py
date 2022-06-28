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

# COMMAND ----------

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size)
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
           epoch=15, strat=None, 
           experiment_id=dais_experiment_id, 
           run_name='baseline_single_tune')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Double Batch Again
# MAGIC 
# MAGIC When we double the batch size again and add 2 more workers it no longer seems to have an effect

# COMMAND ----------

num_workers = 10
pin_memory = True
batch_size = 128

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(data_module, model, 
           num_gpus=1, root_dir=dais_root_folder, 
           epoch=15, strat=None, 
           experiment_id=dais_experiment_id, 
           run_name='baseline_single_tune_large_batch')

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
  epoch=15,
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

# MAGIC %md
# MAGIC 
# MAGIC ### Dual Manual

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64
num_gpus = 2
strat = 'dp'

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(data_module, model, 
           num_gpus=num_gpus, root_dir=dais_root_folder, 
           epoch=15, strat=strat, 
           experiment_id=dais_experiment_id, 
           run_name='opt_dual_tune')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Quad GPU

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64
num_gpus = 4
strat = 'dp'

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(data_module, model, 
           num_gpus=num_gpus, root_dir=dais_root_folder, 
           epoch=15, strat=strat, 
           experiment_id=dais_experiment_id, 
           run_name='opt_quad_tune')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Horovod Runner Dual

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64
num_gpus = 2
#strat = 'dp'

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

hr = HorovodRunner(np=-num_gpus)
hr.run(main_hvd, 
         mlflow_db_host=databricks_host, 
         mlflow_db_token=databricks_host, 
         data_module=data_module, 
         model=model, 
         root_dir=dais_root_folder, 
         epochs=15, 
         run_name='hvd_runner_dual', 
         experiment_id=dais_experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Horovod Runner Quad

# COMMAND ----------

num_workers = 8
pin_memory = True
batch_size = 64
num_gpus = 4
#strat = 'dp'

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

hr = HorovodRunner(np=-num_gpus)
hr.run(main_hvd, 
         mlflow_db_host=databricks_host, 
         mlflow_db_token=databricks_token
         data_module=data_module, 
         model=model, 
         root_dir=dais_root_folder, 
         epochs=15, 
         run_name='hvd_runner_quad', 
         experiment_id=dais_experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Analyse with Tensorboard

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/Users/brian.law@databricks.com/dais_exp/logs

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC experiment_log_dir = '/dbfs/Users/brian.law@databricks.com/dais_exp/logs'

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------


