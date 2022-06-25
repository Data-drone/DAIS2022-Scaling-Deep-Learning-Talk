// Databricks notebook source
// MAGIC %scala
// MAGIC package com.databricks.fieldeng
// MAGIC 
// MAGIC import java.io.File
// MAGIC import scala.io.Source
// MAGIC import java.io._
// MAGIC import org.apache.spark.sql.{Dataset, SparkSession}
// MAGIC import org.apache.spark.sql.functions.{col}
// MAGIC import com.databricks.dbutils_v1.DBUtilsHolder.dbutils
// MAGIC import scala.collection.mutable.{ArrayBuffer, Buffer}
// MAGIC import org.apache.spark.Partitioner
// MAGIC 
// MAGIC case class FileEntry(partition: Int, path: String, size: Long)
// MAGIC case class BlockFileEntry(partition: Int, path: String, size: Long, blockSize: Long)
// MAGIC 
// MAGIC // Class to make org.apache.hadoop.conf.Configuration Serializable
// MAGIC class ConfigSerializable(var conf: org.apache.hadoop.conf.Configuration) extends Serializable {
// MAGIC   def this() {
// MAGIC     this(new org.apache.hadoop.conf.Configuration())
// MAGIC   }
// MAGIC 
// MAGIC   def get(): org.apache.hadoop.conf.Configuration = conf
// MAGIC 
// MAGIC   private def writeObject (out: java.io.ObjectOutputStream): Unit = {
// MAGIC     conf.write(out)
// MAGIC   }
// MAGIC 
// MAGIC   private def readObject (in: java.io.ObjectInputStream): Unit = {
// MAGIC     conf = new org.apache.hadoop.conf.Configuration()
// MAGIC     conf.readFields(in)
// MAGIC   }
// MAGIC 
// MAGIC   private def readObjectNoData(): Unit = {
// MAGIC     conf = new org.apache.hadoop.conf.Configuration()
// MAGIC   }
// MAGIC }
// MAGIC 
// MAGIC object Worker {
// MAGIC     def ls(path: String, recurse: Boolean = false)(implicit sprk: SparkSession): Dataset[String] = {
// MAGIC     import sprk.implicits._
// MAGIC     val df = sprk.createDataset(Seq(1))
// MAGIC     df.mapPartitions(_ => {
// MAGIC       def listFiles(p: String): Iterator[String] = {
// MAGIC         val paths = new File(p).list()
// MAGIC         if( paths != null ) {
// MAGIC           if( recurse ) {
// MAGIC             paths.par.map(lf => (if( p.endsWith("/") ) p else (p + "/")) + lf).iterator ++ paths.par.iterator.flatMap(f => {
// MAGIC               listFiles((if( p.endsWith("/") ) p else (p + "/")) + f)
// MAGIC             })
// MAGIC           }
// MAGIC           else {
// MAGIC             paths.par.map(lf => (if( p.endsWith("/") ) p else (p + "/")) + lf).iterator
// MAGIC           }
// MAGIC         }
// MAGIC         else {
// MAGIC           Iterator.empty
// MAGIC         }
// MAGIC       }
// MAGIC       Iterator(path) ++ listFiles(path)
// MAGIC     })
// MAGIC   }
// MAGIC 
// MAGIC   def cat(path: String)(implicit sprk: SparkSession): Dataset[String] = {
// MAGIC     import sprk.implicits._
// MAGIC     val df = sprk.createDataset(Seq(1))
// MAGIC     df.mapPartitions(_ => {
// MAGIC       def catFile(p: String): String = {
// MAGIC         Source.fromFile(p).getLines.mkString("\n")
// MAGIC       }
// MAGIC       Iterator(catFile(path))
// MAGIC     })
// MAGIC   }
// MAGIC }
// MAGIC 
// MAGIC object Driver {
// MAGIC   class CustomPartitioner(partitions: Int) extends Partitioner {
// MAGIC     def numPartitions: Int = partitions
// MAGIC 
// MAGIC     def getPartition(key: Any): Int = key match {
// MAGIC       case k: Int => k % partitions
// MAGIC     }
// MAGIC   }
// MAGIC   
// MAGIC   private val checkpointFileSuffix = ".last-modified-time.txt"
// MAGIC 
// MAGIC   private val copyFile: (String, String, org.apache.hadoop.conf.Configuration, org.apache.hadoop.fs.FileSystem, org.apache.hadoop.fs.FileSystem, Boolean) => Unit = (src: String, dst: String, conf: org.apache.hadoop.conf.Configuration, srcFs: org.apache.hadoop.fs.FileSystem, dstFs: org.apache.hadoop.fs.FileSystem, deleteSource: Boolean) => {
// MAGIC     val srcPath = new org.apache.hadoop.fs.Path(src)
// MAGIC     val dstPath = new org.apache.hadoop.fs.Path(dst)
// MAGIC     org.apache.hadoop.fs.FileUtil.copy(
// MAGIC       srcFs,
// MAGIC       srcPath,
// MAGIC       dstFs,
// MAGIC       dstPath,
// MAGIC       deleteSource,
// MAGIC       conf
// MAGIC     )
// MAGIC   }
// MAGIC   
// MAGIC   private val deleteFile: (String, org.apache.hadoop.fs.FileSystem, Boolean) => Boolean = (path: String, fs: org.apache.hadoop.fs.FileSystem, recursive: Boolean) => {
// MAGIC     val p = new org.apache.hadoop.fs.Path(path)
// MAGIC     if( recursive ) {
// MAGIC       try {
// MAGIC         fs.delete(p, recursive)
// MAGIC       }
// MAGIC       catch {
// MAGIC         case ex: Exception => println(ex)
// MAGIC         false
// MAGIC       }
// MAGIC     }
// MAGIC     else {
// MAGIC       fs.delete(p, recursive)
// MAGIC     }
// MAGIC   }
// MAGIC 
// MAGIC   private def getPathPrefix(path: String, conf: org.apache.hadoop.conf.Configuration): Option[String] = {
// MAGIC     val p = new org.apache.hadoop.fs.Path(path)
// MAGIC     val iter = p.getFileSystem(conf).listLocatedStatus(p)
// MAGIC     if(iter.hasNext()) {
// MAGIC       val f = iter.next()
// MAGIC       Some(new File(f.getPath().toString).getParent())
// MAGIC     }
// MAGIC     else {
// MAGIC       None
// MAGIC     }
// MAGIC /*    val paths = nondist_ls(path, false).map(_.path)
// MAGIC     if( paths.nonEmpty ) {
// MAGIC       Some(new File(paths.head).getParent())
// MAGIC     }
// MAGIC     else {
// MAGIC       None
// MAGIC     }*/
// MAGIC   }
// MAGIC   
// MAGIC   private def isDir(path: String): Boolean = {
// MAGIC     path.endsWith("/")
// MAGIC   }
// MAGIC   
// MAGIC   private def isDir(path: org.apache.hadoop.fs.Path, fs: org.apache.hadoop.fs.FileSystem): Boolean = {
// MAGIC     fs.getFileStatus(path).isDirectory()
// MAGIC   }
// MAGIC   
// MAGIC   private def isFile(path: String): Boolean = {
// MAGIC     !isDir(path)
// MAGIC   }
// MAGIC 
// MAGIC   private def isFile(path: org.apache.hadoop.fs.Path, fs: org.apache.hadoop.fs.FileSystem): Boolean = {
// MAGIC     !isDir(path, fs)
// MAGIC   }
// MAGIC 
// MAGIC   private def isEmptyDir(path: String, fs: org.apache.hadoop.fs.FileSystem): Boolean = {
// MAGIC     if( isDir(path) ) {
// MAGIC       val p = new org.apache.hadoop.fs.Path(path)
// MAGIC       !fs.listLocatedStatus(p).hasNext()
// MAGIC     }
// MAGIC     else {
// MAGIC       false
// MAGIC     }
// MAGIC   }
// MAGIC   
// MAGIC   private def isEmptyDir(path: org.apache.hadoop.fs.Path, fs: org.apache.hadoop.fs.FileSystem): Boolean = {
// MAGIC     if( fs.getFileStatus(path).isDirectory() ) {
// MAGIC       !fs.listLocatedStatus(path).hasNext()
// MAGIC     }
// MAGIC     else {
// MAGIC       false
// MAGIC     }
// MAGIC   }
// MAGIC   
// MAGIC   private def isSourceModified(path: org.apache.hadoop.fs.Path, metadataTargetPath: Option[String], srcFs: org.apache.hadoop.fs.FileSystem, dstFs: org.apache.hadoop.fs.FileSystem): Boolean = {
// MAGIC     !metadataTargetPath.exists(mp => {
// MAGIC       val mpp = new org.apache.hadoop.fs.Path(mp)
// MAGIC       val fs = dstFs
// MAGIC       if(fs.exists(mpp)) {
// MAGIC         val stream = fs.open(mpp)
// MAGIC         val modTimeStr = scala.io.Source.fromInputStream(stream).mkString
// MAGIC         stream.close()
// MAGIC         if(modTimeStr.nonEmpty) {
// MAGIC           val modTime = modTimeStr.toLong
// MAGIC           modTime == srcFs.getFileStatus(path).getModificationTime()
// MAGIC         }
// MAGIC         else {
// MAGIC           false
// MAGIC         }
// MAGIC       }
// MAGIC       else {
// MAGIC         false
// MAGIC       }
// MAGIC     })
// MAGIC   }
// MAGIC 
// MAGIC   private def updateSourceModified(path: org.apache.hadoop.fs.Path, metadataTargetPath: Option[String], srcFs: org.apache.hadoop.fs.FileSystem, dstFs: org.apache.hadoop.fs.FileSystem): Unit = {
// MAGIC     metadataTargetPath.map(mp => {
// MAGIC       val mpp = new org.apache.hadoop.fs.Path(mp)
// MAGIC       val fs = dstFs
// MAGIC       val stream = fs.create(mpp)
// MAGIC       val pw = new PrintWriter(stream)
// MAGIC       pw.write(srcFs.getFileStatus(path).getModificationTime().toString)
// MAGIC       pw.close()
// MAGIC     })
// MAGIC   }
// MAGIC 
// MAGIC   private def copyFiles(sourcePath: String,
// MAGIC                         destDirs: Array[String],
// MAGIC                         checkpointPath: Option[String],
// MAGIC                         numPartitions: Option[Int] = None,
// MAGIC                         maxPathsPerPartition: Int = 500,
// MAGIC                         maxFileSizePerPartition: Long = 10L*1024*1024*1024,
// MAGIC                         deleteSource: Boolean = false)(implicit sprk: SparkSession): Unit = {
// MAGIC     import sprk.implicits._
// MAGIC     val hadoopConf = new ConfigSerializable(sprk.sessionState.newHadoopConf())
// MAGIC     val pathPrefix = getPathPrefix(sourcePath, hadoopConf.get())
// MAGIC     pathPrefix.map {
// MAGIC       pp => {
// MAGIC         val filesFromLs = dist_ls(ArrayBuffer(sourcePath), true, false, hadoopConf)(sprk)
// MAGIC /*        val files = checkpointPath.flatMap(cp => {
// MAGIC             dbutils.fs.mkdirs(cp)
// MAGIC             val cpPrefix = getPathPrefix(cp)
// MAGIC             cpPrefix.map(cpp => {
// MAGIC               val filesFromCheckpoint = ls(cp, true)
// MAGIC               balanceFiles(filesFromLs.filter(f => !filesFromCheckpoint.exists(fc => {
// MAGIC               val fileName = f.path.substring(pp.length+1)
// MAGIC               val fileNameCp = fc.path.substring(cpp.length+1)
// MAGIC               s"${fileName}${checkpointFileSuffix}" == fileNameCp 
// MAGIC             })), maxPathsPerPartition)
// MAGIC             })
// MAGIC         }).getOrElse(balanceFiles(filesFromLs, maxPathsPerPartition)).map(f => (f.partition, f.path))*/
// MAGIC         val files = balanceFiles(filesFromLs, maxPathsPerPartition, maxFileSizePerPartition)(sprk).map(f => (f.partition, f.path))
// MAGIC         val filesDf = numPartitions.fold({
// MAGIC           val numPart = if(files.nonEmpty) files.par.maxBy(_._1)._1 + 1 else 1
// MAGIC           val partitioner = new CustomPartitioner(numPart)
// MAGIC           sprk.createDataset(files).rdd.partitionBy(partitioner).toDF("part", "path").select(col("path")).as[String]
// MAGIC         })(p => sprk.createDataset(files).toDF("part", "path").repartition(p, col("part")).select(col("path")).as[String])
// MAGIC         destDirs.par.foreach(destDir => {
// MAGIC           sprk.sparkContext.setLocalProperty("spark.scheduler.pool", destDir)
// MAGIC           val destPathC = destDir
// MAGIC           val copyFileC = copyFile
// MAGIC           filesDf.foreachPartition((i: Iterator[String]) => {
// MAGIC             val srcFs = new org.apache.hadoop.fs.Path(sourcePath).getFileSystem(hadoopConf.get())
// MAGIC             val dstFs = new org.apache.hadoop.fs.Path(destPathC).getFileSystem(hadoopConf.get())
// MAGIC             i.foreach(f => {
// MAGIC               val fileName = f.substring(pp.length+1)
// MAGIC               val targetPath = s"${destPathC}${fileName}"
// MAGIC               val metadataTargetPath = checkpointPath.map(cp => s"${cp}${fileName}${checkpointFileSuffix}")
// MAGIC               val srcPath = new org.apache.hadoop.fs.Path(s"$f")
// MAGIC               if( srcFs.exists(srcPath)) {
// MAGIC                 if(isFile(f)) {
// MAGIC                   val isModified = isSourceModified(srcPath, metadataTargetPath, srcFs, dstFs)
// MAGIC                   if(isModified) {
// MAGIC                     copyFileC(f, targetPath, hadoopConf.get(), srcFs, dstFs, deleteSource)
// MAGIC                   }
// MAGIC                   if(deleteSource) {
// MAGIC                     val parentDir = srcPath.getParent()
// MAGIC                     if(isEmptyDir(parentDir, srcFs)) {
// MAGIC                       dstFs.delete(parentDir, true)
// MAGIC                     }
// MAGIC                   }
// MAGIC                   if(isModified) {
// MAGIC                     updateSourceModified(srcPath, metadataTargetPath, srcFs, dstFs)
// MAGIC                   }
// MAGIC                 }
// MAGIC                 else if( isEmptyDir(f, srcFs) ) {
// MAGIC                   val p = new org.apache.hadoop.fs.Path(targetPath)
// MAGIC                   dstFs.mkdirs(p)
// MAGIC                   if(deleteSource) {
// MAGIC                     dstFs.delete(srcPath, true)
// MAGIC                   }
// MAGIC                 }
// MAGIC               }
// MAGIC             })
// MAGIC           })
// MAGIC         })
// MAGIC         if(deleteSource) {
// MAGIC           dbutils.fs.rm(sourcePath, true)
// MAGIC         }
// MAGIC       }
// MAGIC     }
// MAGIC   }
// MAGIC 
// MAGIC   private def deleteFiles(path: String,
// MAGIC                           hadoopConf: ConfigSerializable,
// MAGIC                           numPartitions: Option[Int] = None,
// MAGIC                           filesOnly: Boolean = false,
// MAGIC                           maxPathsPerPartition: Int = 500)(implicit sprk: SparkSession): Unit = {
// MAGIC     import sprk.implicits._
// MAGIC     val srcPath = new org.apache.hadoop.fs.Path(path)
// MAGIC     if( srcPath.getFileSystem(hadoopConf.get()).exists(srcPath) ) {
// MAGIC       val filesFromLs = dist_ls(ArrayBuffer(path), true, filesOnly, hadoopConf)
// MAGIC       val files = balanceFiles(filesFromLs, maxPathsPerPartition, 0)(sprk).map(f => (f.partition, f.path))
// MAGIC       if( files.nonEmpty ) {
// MAGIC         val filesDf = numPartitions.fold({
// MAGIC           val numPart = if(files.nonEmpty) files.par.maxBy(_._1)._1 + 1 else 1
// MAGIC           val partitioner = new CustomPartitioner(numPart)
// MAGIC           sprk.createDataset(files).rdd.partitionBy(partitioner).toDF("part", "path").select(col("path")).as[String]
// MAGIC         })(p => sprk.createDataset(files).toDF("part", "path").repartition(p, col("part")).select(col("path")).as[String])
// MAGIC         val deleteFileC = deleteFile
// MAGIC         filesDf.foreachPartition((i: Iterator[String]) => {
// MAGIC           val fs = new org.apache.hadoop.fs.Path(path).getFileSystem(hadoopConf.get())
// MAGIC           i.foreach(f => deleteFileC(f, fs, !filesOnly))
// MAGIC         })
// MAGIC       }
// MAGIC     }
// MAGIC   }
// MAGIC 
// MAGIC   def balanceFiles(files: Buffer[FileEntry],
// MAGIC                    maxPathsPerPartition: Int = 500,
// MAGIC                    maxFileSizePerPartition: Long = 10L*1024*1024*1024)(implicit sprk: SparkSession): Buffer[FileEntry] = {
// MAGIC     import sprk.implicits._
// MAGIC     if(files.nonEmpty) {
// MAGIC       val blockSize = 100L*1024
// MAGIC       val blockAdjustedFiles = files.map(f => BlockFileEntry(partition=f.partition, path=f.path, size=f.size, blockSize=(f.size/blockSize+1L)*blockSize))
// MAGIC       val maxSize = if( maxFileSizePerPartition > 0 ) Math.min(maxFileSizePerPartition, blockAdjustedFiles.par.maxBy(_.blockSize).blockSize) else if( maxFileSizePerPartition < 0 ) blockAdjustedFiles.par.maxBy(_.blockSize).blockSize else 0
// MAGIC       val (_, _, _, fileEntries) = sprk.createDataset(blockAdjustedFiles).orderBy(col("blockSize")).collect().par.foldLeft((0L, 0, 0, ArrayBuffer.empty[BlockFileEntry])){
// MAGIC         case ((tsize, part, pathsTotal, a), e) =>
// MAGIC           if( (maxSize > 0 && tsize + e.blockSize > maxSize) || (maxPathsPerPartition > 0 && pathsTotal+1 > maxPathsPerPartition) ) {
// MAGIC             (e.blockSize, part+1, 1, a :+ e.copy(partition=part+1))
// MAGIC           }
// MAGIC           else {
// MAGIC             (tsize + e.blockSize, part, pathsTotal+1, a :+ e.copy(partition=part))
// MAGIC           }
// MAGIC       }
// MAGIC       fileEntries.map(f => FileEntry(partition=f.partition, path=f.path, size=f.size))
// MAGIC     }
// MAGIC     else {
// MAGIC       ArrayBuffer.empty[FileEntry]
// MAGIC     }
// MAGIC   }
// MAGIC 
// MAGIC   private def dist_ls(dirs: Buffer[String], recursive: Boolean, filesOnly: Boolean, conf: ConfigSerializable)(implicit sprk: SparkSession): Buffer[FileEntry] = {
// MAGIC     import sprk.implicits._
// MAGIC     val partitioner = new CustomPartitioner(Math.min(dirs.length, 10000))
// MAGIC     val filesDf = sprk.createDataset(dirs.zipWithIndex).toDF("path", "index").select("index", "path").as[(Int, String)].rdd.partitionBy(partitioner).map(_._2).flatMap({
// MAGIC       case p => {
// MAGIC         val pp = new org.apache.hadoop.fs.Path(p)
// MAGIC         val iter = if( filesOnly ) pp.getFileSystem(conf.get()).listFiles(pp, false) else pp.getFileSystem(conf.get()).listLocatedStatus(pp)
// MAGIC         var files = ArrayBuffer.empty[FileEntry]
// MAGIC         while(iter.hasNext()) {
// MAGIC           val f = iter.next()
// MAGIC           files = files :+ FileEntry(partition = 0, path = s"""${f.getPath().toString}${if(f.isDirectory()) "/" else ""}""", size = f.getLen())
// MAGIC         }
// MAGIC         files
// MAGIC       }
// MAGIC     })
// MAGIC     val files = filesDf.collect().toBuffer
// MAGIC     if( recursive ) {
// MAGIC       val dirChildren = files.filter(f => f.path.endsWith("/"))
// MAGIC       if(dirChildren.nonEmpty) {
// MAGIC         files ++ dist_ls(dirChildren.map(_.path), recursive, filesOnly, conf)(sprk)
// MAGIC       }
// MAGIC       else {
// MAGIC         files
// MAGIC       }
// MAGIC     }
// MAGIC     else {
// MAGIC       files
// MAGIC     }
// MAGIC   }
// MAGIC 
// MAGIC   def ls(path: String,
// MAGIC          recursive: Boolean = true,
// MAGIC          filesOnly: Boolean = false,
// MAGIC          distributed: Boolean = true)(implicit sprk: SparkSession): Buffer[FileEntry] = {
// MAGIC     if( distributed ) {
// MAGIC       dist_ls(path, recursive, filesOnly)(sprk)
// MAGIC     }
// MAGIC     else {
// MAGIC       nondist_ls(path, recursive, filesOnly)
// MAGIC     }
// MAGIC   }
// MAGIC 
// MAGIC   def dist_ls(path: String, recursive: Boolean = true, filesOnly: Boolean = false)(implicit sprk: SparkSession): Buffer[FileEntry] = {
// MAGIC     import sprk.implicits._
// MAGIC     val hadoopConf = new ConfigSerializable(sprk.sessionState.newHadoopConf())
// MAGIC     dist_ls(ArrayBuffer(path), recursive, filesOnly, hadoopConf)(sprk)
// MAGIC   }
// MAGIC 
// MAGIC   def nondist_ls(path: String, recursive: Boolean = true, filesOnly: Boolean = false): Buffer[FileEntry] = {
// MAGIC     dbutils.fs.ls(path).par.flatMap(fi => {
// MAGIC       if( recursive && fi.name.endsWith("/") ) {
// MAGIC         val dirFiles = nondist_ls(fi.path, recursive, filesOnly)
// MAGIC         if( filesOnly ) {
// MAGIC           dirFiles
// MAGIC         }
// MAGIC         else {
// MAGIC           dirFiles :+ FileEntry(0, fi.path, fi.size)
// MAGIC         }
// MAGIC       }
// MAGIC       else {
// MAGIC         ArrayBuffer(FileEntry(0, fi.path, fi.size))
// MAGIC       }
// MAGIC     }).toBuffer
// MAGIC   }
// MAGIC   
// MAGIC   def explode_cp(sourcePath: String,
// MAGIC                  destPaths: Array[String],
// MAGIC                  numPartitions: Option[Int] = None,
// MAGIC                  maxPathsPerPartition: Int = 500,
// MAGIC                  maxFileSizePerPartition: Long = 10L*1024*1024*1024)(implicit sprk: SparkSession): Unit = {
// MAGIC     copyFiles(sourcePath, destPaths, None, numPartitions, maxPathsPerPartition, maxFileSizePerPartition, false)
// MAGIC   }
// MAGIC 
// MAGIC   def cp(sourcePath: String,
// MAGIC          destPath: String,
// MAGIC          checkpointPath: String = "",
// MAGIC          numPartitions: Option[Int] = None,
// MAGIC          maxPathsPerPartition: Int = 500,
// MAGIC          maxFileSizePerPartition: Long = 10L*1024*1024*1024,
// MAGIC          deleteSource: Boolean = false)(implicit sprk: SparkSession): Unit = {
// MAGIC     val checkpointOpt = Option(checkpointPath).filter(_.trim.nonEmpty)
// MAGIC     checkpointOpt.foreach(cp => {
// MAGIC       val cpp = new org.apache.hadoop.fs.Path(cp)
// MAGIC       val sp = new org.apache.hadoop.fs.Path(sourcePath)
// MAGIC       val dp = new org.apache.hadoop.fs.Path(destPath)
// MAGIC       def differentLeftPath(left: org.apache.hadoop.fs.Path, right: org.apache.hadoop.fs.Path): Boolean = {
// MAGIC         var p = left
// MAGIC         while(p != null) {
// MAGIC           if( p.toString == right.toString ) {
// MAGIC             return false
// MAGIC           }
// MAGIC           p = p.getParent()
// MAGIC         }
// MAGIC         true
// MAGIC       }
// MAGIC       def differentPaths(left: org.apache.hadoop.fs.Path, right: org.apache.hadoop.fs.Path): Boolean = {
// MAGIC         if( !differentLeftPath(left, right) ) {
// MAGIC           false
// MAGIC         }
// MAGIC         else {
// MAGIC           differentLeftPath(right, left)
// MAGIC         }
// MAGIC       }
// MAGIC       if( !differentPaths(cpp, dp) ) {
// MAGIC         throw new Exception(s"Checkpoint ${cpp.toString} should not be within destination path ${dp.toString}")
// MAGIC       }
// MAGIC       else if( !differentPaths(cpp, sp) ) {
// MAGIC         throw new Exception(s"Checkpoint ${cpp.toString} should not be within source path ${sp.toString}")
// MAGIC       }
// MAGIC     })
// MAGIC     copyFiles(sourcePath, Array(destPath), checkpointOpt, numPartitions, maxPathsPerPartition, maxFileSizePerPartition, deleteSource)
// MAGIC   }
// MAGIC   
// MAGIC   def rm(path: String,
// MAGIC          numPartitions: Option[Int] = None,
// MAGIC          maxPathsPerPartition: Int = 500)(implicit sprk: SparkSession): Boolean = {
// MAGIC     val srcPath = new org.apache.hadoop.fs.Path(path)
// MAGIC     val hadoopConf = new ConfigSerializable(sprk.sessionState.newHadoopConf())
// MAGIC     if( srcPath.getFileSystem(hadoopConf.get()).exists(srcPath) ) {
// MAGIC       deleteFiles(path, hadoopConf, numPartitions, true, maxPathsPerPartition)
// MAGIC       deleteFiles(path, hadoopConf, numPartitions, false, maxPathsPerPartition)
// MAGIC       if( srcPath.getFileSystem(hadoopConf.get()).exists(srcPath) ) {
// MAGIC         dbutils.fs.rm(path, true)
// MAGIC       }
// MAGIC       else {
// MAGIC         true
// MAGIC       }
// MAGIC     }
// MAGIC     else {
// MAGIC       false
// MAGIC //      throw new Exception(s"Path $path does not exist")
// MAGIC     }
// MAGIC   }
// MAGIC }
