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
