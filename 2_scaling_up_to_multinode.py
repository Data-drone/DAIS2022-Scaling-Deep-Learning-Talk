# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Scaling up with Multiple Nodes
# MAGIC 
# MAGIC Now that we have maxed out a single node, we can move to multinode training

# COMMAND ----------

import horovod.torch as hvd

from TrainLoop.pl_train import main_hvd, build_trainer, main_train
import os

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup Experiment
# MAGIC 
# MAGIC These flags are the same as last time

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

from Models.resnet import ResnetClassification
from Dataloaders.imagenette import ImagenetteDataModule

# COMMAND ----------

data_module = ImagenetteDataModule(imagenette_data_path)
model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes, pretrain=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Trigger runs
# MAGIC 
# MAGIC We use horovod spark to scale up across nodes.
# MAGIC This is because HorovodRunner currently doesn't distribute Python Modules onto worker nodes.
# MAGIC 
# MAGIC *NOTE* The spark error doesn't affect actual model training. It seems that the spark process is just waiting for some output before it terminates and it isn't getting it so it times out

# COMMAND ----------

import horovod.spark 

# COMMAND ----------

66# set to the number of workers * ?num gpu per worker?
num_processes = 2
epochs = 15

model = horovod.spark.run(main_hvd, 
                          kwargs={'mlflow_db_host': databricks_host, 
         'mlflow_db_token': databricks_token, 
         'data_module': data_module, 
         'model': model, 
         'root_dir': dais_root_folder, 
         'epochs': epochs, 
         'run_name': 'hvd_spark_2_wrk', 
         'experiment_id': dais_experiment_id},
         num_proc=num_processes, 
         verbose=2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Moving to 8 GPUS

# COMMAND ----------

66# set to the number of workers * ?num gpu per worker?
num_processes = 8
epochs = 15

model = horovod.spark.run(main_hvd, 
                          kwargs={'mlflow_db_host': databricks_host, 
         'mlflow_db_token': databricks_token, 
         'data_module': data_module, 
         'model': model, 
         'root_dir': dais_root_folder, 
         'epochs': epochs, 
         'run_name': 'hvd_spark_8_wrk', 
         'experiment_id': dais_experiment_id},
         num_proc=num_processes, 
         verbose=2)

# COMMAND ----------


