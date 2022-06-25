// Databricks notebook source
// MAGIC %run ./Utils

// COMMAND ----------

import com.databricks.fieldeng._

// COMMAND ----------

val source = "s3a://performance-assurance-sftp/psm_untar/lte"
val dest = "s3a://performance-assurance-sftp/input2"
val checkpoint = "s3a://performance-assurance-sftp/dest-cp"

// COMMAND ----------

Driver.rm(dest)(spark)
Driver.rm(checkpoint)(spark)

// COMMAND ----------

Driver.rm(dest)(spark)

// COMMAND ----------

Driver.cp(source, dest)(spark)

// COMMAND ----------

val srcs = (0 until 10)
val sites = (0 until 150)
val dests = sites.flatMap(site => srcs.map(src => s"${dest}/${site}_${src}")).toArray
Driver.explode_cp(source, dests)(spark)
//Driver.cp(source, s"${dest}/${site}_${src}")(spark)

// COMMAND ----------

val df = spark.read.format("delta").load(dest)
display(df)
