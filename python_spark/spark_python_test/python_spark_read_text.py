# coding:utf-8

'''
使用spark做word count
'''

from operator import add
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("spark use python").getOrCreate()
# line = spark.read.text('/home/dushao/')
# sparkContext = SparkContext(appName="PythonWordCount")
lines = spark.read.text('/home/dushao/work/python3.6/python_spark/spark_python_test/word') \
    .rdd.map(lambda r: r[0])

counts = lines.flatMap(lambda x: x.split(' ')) \
    .map(lambda x: (x, 1)) \
    .reduceByKey(add)

output = counts.collect()
for (word, count) in output:
    print("%s: %i" % (word, count))

spark.stop()