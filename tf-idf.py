from __future__ import print_function
import sys
import pyspark
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, ArrayType, DoubleType, FloatType
from pyspark.ml.feature import StringIndexer, HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline, PipelineModel



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: vt-w2v.py <input> <w2v-output>", file=sys.stderr)
        exit(-1)
        
    spark = SparkSession.builder.appName('Create tf-idf').getOrCreate()
    
    data = spark.read.load(sys.argv[1])
    
    df = data.filter((col('date') >= '1895') & (col('seq') =='1')) \
            .select(year('date').alias('year'), 'id', 'text')
    
    # https://danvatterott.com/blog/2018/07/08/aggregating-sparse-and-dense-vectors-in-pyspark/
    def dense_to_array(v):
        new_array = list([float(x) for x in v])
        return new_array

    dense_to_array_udf = udf(dense_to_array, ArrayType(FloatType()))

    indexer = StringIndexer(inputCol="id", outputCol="label")
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    vectorizer = CountVectorizer(inputCol="tokens", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="vector", minDocFreq=1)

    pipeline = Pipeline(stages=[indexer, tokenizer, vectorizer, idf])
    model = pipeline.fit(df)

    results = model.transform(df) \
        .select(year('date').alias('year'), 'label', 'vector') \
        .withColumn('vector', dense_to_array_udf('vector'))

    results = model.transform(df).select('year', 'label', 'vector')
    
    results.write \
        .partitionBy('year') \
        .format('csv') \
        .options(compression='gzip', sep='\t', header='true') \
        .save(sys.argv[2])
    
    spark.stop()