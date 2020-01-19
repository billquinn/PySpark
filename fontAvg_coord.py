from __future__ import print_function
import sys
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import col, year, explode, lit, pow
from pyspark.sql.types import LongType


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: fontAvg_coord.py <input> <output>", file=sys.stderr) # sys.stderr= write to file object
        exit(-1)
        
    spark = SparkSession.builder.appName('Create Page Documents').getOrCreate()
    
    num_places = 0
    m = pow(lit(10), num_places).cast(LongType())
    
    df = spark.read.load(sys.argv[1])
    
    # dff = df.filter(col('date') > '1890')
    dff = df.filter((col('date') > '1870') & (col('date') < '1880'))
    
    dff.select(year('date').alias('year'), explode('pages').alias('pages')) \
        .select('year', 'pages.*') \
        .select('year','width','height', explode('regions').alias('regions')) \
        .select('year','width','height', 'regions.*') \
        .select('year','width','height', 'coords.*') \
        .withColumn("xCoord", (col('x') / col('width') * 100 * m).cast(LongType()) / m) \
        .withColumn("yCoord", (col('y') / col('height') * 100 * m).cast(LongType()) / m) \
        .groupby('year', 'xCoord', 'yCoord') \
        .agg({'h':'avg'}) \
        .write \
        .partitionBy('year') \
        .format('csv') \
        .options(compression='gzip', sep='\t', header='true') \
        .save(sys.argv[2])
    
    spark.stop()
