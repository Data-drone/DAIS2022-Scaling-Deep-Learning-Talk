# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Loading Data via Python Module

# Working on the Cloud and Working on prem is different

# COMMAND ----------

local_disk_tmp_dir = '/local_disk0/tmp_data'
username = "brian.law@databricks.com"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TorchVision Example

# COMMAND ----------

tv_dataset_name = 'cifar100'

torchvision_loader = """

import torchvision

CIFAR100 = torchvision.datasets.CIFAR100(
          root='{0}',
          download=True)

""".format(local_disk_tmp_dir)

tv_load_path = "/Users/{0}/test_script/preload_{1}.py".format(username, tv_dataset_name)

dbutils.fs.put(tv_load_path, torchvision_loader, True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## HuggingFace example

# COMMAND ----------

hf_dataset_name = 'imagenet-1k'

hugging_face_loader = """

from datasets import load_dataset

auth_token = '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'
dataset = load_dataset("imagenet-1k",
                       cache_dir='{0}',
                       split=['train', 'validation'],
                       use_auth_token=auth_token)
                       
""".format(local_disk_tmp_dir)

hf_load_path = "/Users/{0}/test_script/preload_hf_{1}.py".format(username, hf_dataset_name)

dbutils.fs.put(hf_load_path, hugging_face_loader, True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Init Script to run on nodes

# COMMAND ----------

def create_init_script(username: str, load_path: str, dataset_name: str):
  init_script = """

  #!/bin/bash

  /databricks/python/bin/pip install datasets

  /databricks/python/bin/python /dbfs{1}



  """.format(username, load_path)

  print(init_script)
  
  init_script_path = "dbfs:/Users/{0}/init/preload_{1}.sh".format(username, dataset_name)

  dbutils.fs.put(init_script_path, init_script, True)
  
  return init_script_path

# COMMAND ----------

tv_load = create_init_script(username, tv_load_path, tv_dataset_name)
#hf_load = create_init_script(username, hf_load_path, hf_dataset_name)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cat /dbfs/Users/brian.law@databricks.com/test_script/preload_cifar100.py

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Triggering the job along with the copy notebook

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC databricks jobs create --help

# COMMAND ----------

HOST = "https://e2-demo-tokyo.cloud.databricks.com/"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

from databricks_cli import sdk
from databricks_cli.jobs.api import JobsApi
import json

client = sdk.ApiClient(host=HOST, token=TOKEN)

jobs = JobsApi(api_client = client)

# COMMAND ----------

target_path = "/Repos/brian.law@databricks.com/scaling_deep_learning/CopyFiles_Notebook"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Job Definition

# COMMAND ----------

hf_job_config ={
        "name": "Copy Files Deep Learning - HF",
        "timeout_seconds": 0,
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "task_key": "CopyFiles_DeepLearning",
                "notebook_task": {
                    "notebook_path": target_path
                },
                "job_cluster_key": "copy_files_temp",
                "timeout_seconds": 0,
                "email_notifications": {}
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "copy_files_temp",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                    "aws_attributes": {
                        "zone_id": "ap-northeast-1a",
                        "first_on_demand": 4,
                        "availability": "SPOT_WITH_FALLBACK",
                        "spot_bid_price_percent": 100,
                        "ebs_volume_type": "GENERAL_PURPOSE_SSD",
                        "ebs_volume_count": 1,
                        "ebs_volume_size": 750
                    },
                    "node_type_id": "m5.16xlarge",
                    "init_scripts": [{
                      "dbfs": {
                        "destination": hf_load
                        }
                      }],
                    "enable_elastic_disk": False,
                    "runtime_engine": "STANDARD",
                    "autoscale": {
                        "min_workers": 2,
                        "max_workers": 30
                    }
                }
            }
        ]
}

# COMMAND ----------

tv_job_config ={
        "name": "Copy Files Deep Learning - TV",
        "timeout_seconds": 0,
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "task_key": "CopyFiles_DeepLearning",
                "notebook_task": {
                    "notebook_path": target_path
                },
                "job_cluster_key": "copy_files_temp",
                "timeout_seconds": 0,
                "email_notifications": {}
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "copy_files_temp",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                    "aws_attributes": {
                        "zone_id": "ap-northeast-1a",
                        "first_on_demand": 4,
                        "availability": "SPOT_WITH_FALLBACK",
                        "spot_bid_price_percent": 100,
                        "ebs_volume_type": "GENERAL_PURPOSE_SSD",
                        "ebs_volume_count": 1,
                        "ebs_volume_size": 400
                    },
                    "node_type_id": "m4.4xlarge",
                    "init_scripts": [{
                      "dbfs": {
                        "destination": tv_load
                        }
                      }],
                    "enable_elastic_disk": False,
                    "runtime_engine": "STANDARD",
                    "autoscale": {
                        "min_workers": 2,
                        "max_workers": 30
                    }
                }
            }
        ]
}

# COMMAND ----------

job_create_id = jobs.create_job(json=hf_job_config)
job_create_id

# COMMAND ----------

job_create_id = jobs.create_job(json=tv_job_config)
job_create_id

# COMMAND ----------

jobs.list_jobs()

# COMMAND ----------

notebook_params={'source_path': 'file:'+local_disk_tmp_dir, 
                 'destination': 'dbfs:/Users/{0}/data/'.format(username), 
                 'checkpoint': 'dbfs:/Users/{0}/tmp_checkpoint'.format(username)}
notebook_params

# COMMAND ----------

jobs.run_now(job_id=job_create_id['job_id'], jar_params=None, notebook_params=notebook_params, 
             python_params=None, spark_submit_params=None, python_named_params=None)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC - Running HF: {'run_id': 313589, 'number_in_job': 313589}
# MAGIC - Running TV: {'run_id': 314616, 'number_in_job': 314616}

# COMMAND ----------

notebook_params={'source_path': 'file:'+local_disk_tmp_dir, 
                 'destination': 'dbfs:/Users/{0}/data/'.format(username), 
                 'checkpoint': 'dbfs:/Users/{0}/tmp_checkpoint'.format(username)}
notebook_params

# COMMAND ----------

cleanup = False

if cleanup:
  jobs.delete_job(job_id=job_create_id['job_id'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Inspect Data Folder

# COMMAND ----------

dbutils.fs.ls('dbfs:/Users/brian.law@databricks.com/data/cifar-100-python/train/')

# COMMAND ----------


