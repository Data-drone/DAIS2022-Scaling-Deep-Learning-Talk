# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Loading Data via Python Module
# MAGIC 
# MAGIC Working on the Cloud and Working on prem is different. Unzipping Folders on the cloud is not efficient and handling even small zip folders can take a long time.
# MAGIC 
# MAGIC There are a few things that we can use to make this a bit easier.
# MAGIC 
# MAGIC All databricks nodes have a /local_disk0 which is a direct attached storage path.
# MAGIC We can download zip files here first and run the unzip here.
# MAGIC 
# MAGIC Once the files have been unzipped we can use a distributed copy in order copy the data across.
# MAGIC Now in order for distributed copy with Spark to work we need the dataset to be present on local_disk0 on each node. So we will need to use an init script in order to make sure that the data is present on all nodes 
# MAGIC 
# MAGIC In this notebook we will:
# MAGIC 
# MAGIC - Create the python script to download and unzip a dataset
# MAGIC - Create an init script to unzip the data on local_disk0
# MAGIC - Create a databricks job to copy the data from local_disk0 to the destination object folder of our choice
# MAGIC 
# MAGIC This solution is better suited for web data sources. In a production scenario there are other tools and process that could be used to manage data ingest.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Configuration Parameters

# COMMAND ----------

local_disk_tmp_dir = '/local_disk0/tmp_data' # temporary cache data for data on local disk
username = "brian.law@databricks.com" # I am storing data in my user folder

## This is the notebook that will run our distcp
### The easiest way to get this address is to create the Workflow manually and check
target_notebook_path = "/Repos/brian.law@databricks.com/DAIS2022-Scaling-Deep-Learning-Talk/Utils/copy_files_nb"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TorchVision Example
# MAGIC 
# MAGIC Here we have an example about a python script that loads a Torchvision Dataset

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
# MAGIC 
# MAGIC Here we construct the init script that we will start out clusters with.
# MAGIC It needs to install any libraries necessary (torchvision is bundled by default)
# MAGIC Then it will run the python script we created above

# COMMAND ----------

def create_init_script(username: str, load_path: str, dataset_name: str) -> str:
  """
  
  This function creates the init script we need and writes it to my dbfs Users folder
  
  """
  
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

# MAGIC %md
# MAGIC 
# MAGIC ## Cross check what we saved out

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cat /dbfs/Users/brian.law@databricks.com/test_script/preload_cifar100.py

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Triggering the job along with the copy notebook
# MAGIC 
# MAGIC We will use API calls to create our Databricks Workflow.
# MAGIC 
# MAGIC See:
# MAGIC - https://docs.databricks.com/data-engineering/jobs/index.html
# MAGIC - https://docs.databricks.com/dev-tools/cli/index.html

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC databricks jobs create --help

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The host is our workspace, we can check this but looking at the url of our workspace.
# MAGIC 
# MAGIC API calls need to have a valid token to run. This can either be a user token or the default one generated for our running notebook.
# MAGIC The `dbutils` 

# COMMAND ----------

HOST = "https://e2-demo-tokyo.cloud.databricks.com/"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

from databricks_cli import sdk
from databricks_cli.jobs.api import JobsApi
import json

client = sdk.ApiClient(host=HOST, token=TOKEN)

jobs = JobsApi(api_client = client)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Job Definition
# MAGIC 
# MAGIC To create a job via API we need to create the config as a dict object
# MAGIC - See: https://docs.databricks.com/dev-tools/api/latest/jobs.html#operation/JobsCreate
# MAGIC 
# MAGIC We can see that the job has `tasks` which in this case is our 

# COMMAND ----------

generic_job_config = {
        "name": "Copy Files Deep Learning",
        "timeout_seconds": 0,
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "task_key": "CopyFiles_DeepLearning",
                "notebook_task": {
                    "notebook_path": target_notebook_path
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
                        "destination": 'REPLACE_ME'
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

# We can then alter the bits we need to for the job name and also to point to the right init_script

tv_job_config = generic_job_config
tv_job_config['name'] = "Copy Files Deep Learning - TV"
tv_job_config['job_clusters'][0]['new_cluster']['init_scripts'][0]['dbfs']['destination'] = tv_load
print(tv_job_config)

# COMMAND ----------

tv_job_create_id = jobs.create_job(json=tv_job_config)
tv_job_create_id

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Check Jobs and Execute Run
# MAGIC 
# MAGIC The Dist copy notebook use widgets so that we can customise the source and destination folder for our copy job.
# MAGIC - See: https://docs.databricks.com/notebooks/widgets.html

# COMMAND ----------

# show all jobs
jobs.list_jobs()

# COMMAND ----------

notebook_params={'source_path': 'file:'+local_disk_tmp_dir, 
                 'destination': 'dbfs:/Users/{0}/data/'.format(username), 
                 'checkpoint': 'dbfs:/Users/{0}/tmp_checkpoint'.format(username)}
notebook_params

# COMMAND ----------

# this will trigger the job with the notebook_params that we have set
jobs.run_now(job_id=job_create_id['job_id'], jar_params=None, notebook_params=notebook_params, 
             python_params=None, spark_submit_params=None, python_named_params=None)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Post Jobs Cleanup
# MAGIC 
# MAGIC This is just housekeeping for our example

# COMMAND ----------

cleanup = False

if cleanup:
  jobs.delete_job(job_id=job_create_id['job_id'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Inspect Data Folder
# MAGIC 
# MAGIC We can check what data is in the folder to verify everything worked

# COMMAND ----------

dbutils.fs.ls('dbfs:/Users/brian.law@databricks.com/data/')

# COMMAND ----------


