# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Scaling on Single Node
# MAGIC 
# MAGIC Aside from tuning batches, memory and scaling up on GPUs there are also increasingly advanced algorithms that can help speed up execution speed

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

databricks_token

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
# MAGIC ## Baseline Model
# MAGIC 
# MAGIC This current one maxes out the single GPU already

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
# MAGIC ## Deepspeed

# COMMAND ----------


