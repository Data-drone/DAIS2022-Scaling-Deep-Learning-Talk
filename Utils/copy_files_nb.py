# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Set Up Widgets

# COMMAND ----------

username = 'brian.law@databricks.com'

# COMMAND ----------

dbutils.widgets.text('source_path', 'file:/local_disk0/tmp_data')
dbutils.widgets.text('destination', 'dbfs:/Users/{0}/data/')
dbutils.widgets.text('checkpoint', 'dbfs:/Users/{0}/tmp_checkpoint'.format(username))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Load the Libs

# COMMAND ----------

# MAGIC %run ./distcp/Utils

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC import com.databricks.fieldeng._

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Run Copy

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val source_path = dbutils.widgets.get("source_path")
# MAGIC val destination = dbutils.widgets.get("destination")
# MAGIC val checkpoint = dbutils.widgets.get("checkpoint")

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC Driver.rm(checkpoint)(spark)
# MAGIC 
# MAGIC Driver.cp(source_path, destination, checkpoint)(spark)
