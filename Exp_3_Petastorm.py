# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Reloading Data into Petastorm
# MAGIC 
# MAGIC Lets reformat the image folder files into a delta table
# MAGIC 
# MAGIC We will:
# MAGIC - Read the train data
# MAGIC - Read the val data
# MAGIC - Create numerical labels which is what we need to feed into the model

# COMMAND ----------

import os
from petastorm.spark import SparkDatasetConverter, make_spark_converter

CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

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
# MAGIC ## Reformatting the data 

# COMMAND ----------

train_df = spark.read.format("image") \
            .load('/Users/brian.law@databricks.com/data/imagenette2/train/*')

val_df = spark.read.format("image") \
            .load('/Users/brian.law@databricks.com/data/imagenette2/val/*')


# COMMAND ----------

from pyspark.sql import functions as F

train_df = train_df.select('image', F.col('image.origin').alias('path')) \
  .withColumn('image_class', F.element_at(F.split(F.col('path'), '/'), -2) ) \
  .withColumn('split', F.lit('train'))

val_df = val_df.select('image', F.col('image.origin').alias('path')) \
  .withColumn('image_class', F.element_at(F.split(F.col('path'), '/'), -2) ) \
  .withColumn('split', F.lit('val'))

print('train size {0}, val size {1}'.format(train_df.count(), val_df.count() ))

unified_df = train_df.union(val_df)

print('unified size {0}'.format(unified_df.count()))

# COMMAND ----------

import pyspark.sql.types as T

# Convert string to numeric
classes = list(unified_df.select("image_class").distinct().toPandas()["image_class"])

def to_class_index(class_name:str):
  """
  Converts classes to a class_index so that we can create a tensor object
  """
  
  return classes.index(class_name)

class_index = udf(to_class_index, T.LongType())  # PyTorch loader required a long

preprocessed_df = unified_df.withColumn("label", class_index(F.col("image_class")))

display(preprocessed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create Lakehouse Tables
# MAGIC 
# MAGIC Here we will create and optimize the lakehouse table for our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS brian_imagenette

# COMMAND ----------

preprocessed_df.write.mode('overwrite').saveAsTable('brian_imagenette.full_dataset')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC OPTIMIZE brian_imagenette.full_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Check the dataset and test out the dataloader
# MAGIC 
# MAGIC Need to convert labels into num
# MAGIC Need

# COMMAND ----------

train_df = spark.sql("""
SELECT image.data as content, label FROM brian_imagenette.full_dataset
WHERE split = 'train'
""")

val_df = spark.sql("""
SELECT image.data as content, label FROM brian_imagenette.full_dataset
WHERE split = 'val'
""") 

# COMMAND ----------

train_converter = make_spark_converter(train_df)
val_converter = make_spark_converter(val_df)

# COMMAND ----------

from Dataloaders.imagenette_petastorm import ImagenettePetaDataModule
from Models.resnet import ResnetClassification
from TrainLoop.pl_train import main_hvd, build_trainer, main_train


datamodule = ImagenettePetaDataModule(train_converter=train_converter, 
                               val_converter=val_converter,
                                     batch_size=128)

# COMMAND ----------

model = ResnetClassification(*[3, 224, 224], num_classes=10, pretrain=False)

# COMMAND ----------

## It is important to set the root dir for Repos as we cannot write to the local folder with code
main_train(datamodule, model, 
           num_gpus=1, root_dir=dais_root_folder, 
           epoch=15, strat=None, 
           experiment_id=dais_experiment_id, 
           run_name='baseline_petastorm')

# COMMAND ----------


