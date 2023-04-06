import pyspark
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

# Create spark session

spark = SparkSession.builder.appName("ParquetPartitioner").getOrCreate()

# Read parquet file

df_pyspark = spark.read.parquet("diabetes.parquet")

# Create snappy parquets in data folder

df_pyspark.write.parquet("data/", compression="snappy",partitionBy="Age")

# Control Number of Records per Partition File

df_pyspark.write.option("maxRecordsPerFile", 2).parquet("data/", compression="snappy",partitionBy="Age")

# Partition by Records size

df_pyspark.write.option("maxRecordsPerFile", 2).parquet("data/", compression="snappy")

# Read snappy files from folders

df_pyspark2 = spark.read.parquet("data/")

# examples

# https://sparkbyexamples.com/pyspark/pyspark-read-and-write-parquet-file/
# https://kontext.tech/article/1173/pyspark-partitionby-with-examples
# https://mageswaran1989.medium.com/a-dive-into-apache-spark-parquet-reader-for-small-file-sizes-fabb9c35f64e

# Read parquet with pandas

data = pd.read_parquet("filename.parquet",engine ="pyarrow")

# Partition of data

data.to_parquet("./folder", compression = "snappy", partition_cols = ["Month"])

# reading partitioned data

pd.read_parquet("./folder", engine = "pyarrow", filters =[("Month",">","2")])
