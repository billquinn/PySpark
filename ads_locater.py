from __future__ import print_function
import sys
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import col, year, explode, lit, pow
from pyspark.sql.types import LongType


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: ads_locater.py <input> <output>", file=sys.stderr) # sys.stderr= write to file object
        exit(-1)
        
    spark = SparkSession.builder.appName('Create Page Documents').getOrCreate()
    
    num_places = 0
    m = pow(lit(10), num_places).cast(LongType())
    
    df = spark.read.load(sys.argv[1])
    
    # dff = df.filter(col('date') > '1890')
    dff = df.filter((col('date') > '1866') & (col('id').rlike('.*sn83030214.*')))
    
    dff.select(year('date').alias('year'), explode('pages').alias('pages')) \
        .select('year', 'pages.*') \
        .select('year','seq','height', explode('regions').alias('regions')) \
        .select('year', 'seq', 'regions.*') \
        .select('year', 'seq', 'coords.*') \
        .groupby('year', 'seq') \
        .agg({'h':'avg'}) \
        .write \
        .partitionBy('year') \
        .format('csv') \
        .options(compression='gzip', sep='\t', header='true') \
        .save(sys.argv[2])
    
    spark.stop()
