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
# MAGIC 
# MAGIC We explicity provide the workspace and token for mlflow because it doesn't necessarily get picked up by the separate Python Processes that parallel training spins off. 
# MAGIC 
# MAGIC For these examples we use the imagenette dataset.
# MAGIC See:
# MAGIC - https://github.com/fastai/imagenette

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
# MAGIC 
# MAGIC We put our main modelling code into a standard python module here.
# MAGIC See:
# MAGIC - https://databricks.com/blog/2021/10/07/databricks-repos-is-now-generally-available.html

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
# MAGIC 
# MAGIC We set a baseline experiment first so that we have something to benchmark against and work off of. This is a single GPU setup
# MAGIC 
# MAGIC This took 34 mins to run with 15 epochs in my test setup with AWS g4 node

# COMMAND ----------

batch_size = 32

# COMMAND ----------

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size)
model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(data_module, model, 
           num_gpus=1, root_dir=dais_root_folder, 
           epoch=15, strat=None, 
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
# MAGIC 
# MAGIC On my test setup with g4 node this run took 30 mins with 15 epochs

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
# MAGIC When we double the batch size again and add 2 more workers it no longer seems to have an effect.
# MAGIC 
# MAGIC This run took 30 mins with 15 epochs as well 

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
           epoch=3, strat=None, 
           experiment_id=dais_experiment_id, 
           run_name='baseline_single_tune_large_batch')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Adding GPUs
# MAGIC 
# MAGIC We have now hit the limit on what we can get tuning the Dataloader. So lets look to start adding GPUs to see the effect.
# MAGIC 
# MAGIC *Note* I use 'dp' here just to quick illustrate dual gpu.
# MAGIC PyTorch does not recommend using 'dp' usually but the recommended option 'ddp' won't work in an interactive notebook. We will address this later

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Dual GPU with DP
# MAGIC 
# MAGIC This run took 21 mins with 15 epochs. So we can see that we definitely aren't achieving much in the way of scaling

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
# MAGIC 
# MAGIC Moving to quad GPU doesn't give us much in the way of speed ups either.
# MAGIC 
# MAGIC 15 epochs took 20 minutes

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
# MAGIC 
# MAGIC Before I mentioned that 'dp' is not recommended. In order to be able to leverage a more performant distribution methodology we will use HorovodRunner. 
# MAGIC 
# MAGIC See:
# MAGIC - https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html
# MAGIC 
# MAGIC With HorovodRunner 15 epochs takes 17 minutes now we are starting to get decent scaling for our job

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

# negative number will make Horovod run on the driver node
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
# MAGIC 
# MAGIC Using HorovodRunner with 4 gpus took 10 minutes with 15 epochs
# MAGIC 
# MAGIC Finally we can scale our code more efficiently

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
# MAGIC 
# MAGIC Databricks supports Tensorboard which can be run in a notebook or opened up on a different tab
# MAGIC 
# MAGIC See: 
# MAGIC - https://docs.databricks.com/applications/machine-learning/train-model/tensorflow.html

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


