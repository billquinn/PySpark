from __future__ import print_function
import sys
import pyspark
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec, Word2VecModel
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import *
from pyspark.sql.types import *


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: vt-w2v.py <input> <w2v-output>", file=sys.stderr)
        exit(-1)
        
    spark = SparkSession.builder.appName('Create w2v').getOrCreate()
    
    df = spark.read.load(sys.argv[1])
    
    dff = df.filter(col('date') > '1895')
    
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="words")
    word2vec = Word2Vec(vectorSize=200, minCount=100, inputCol="words", outputCol="vectors")

    pipeline = Pipeline(stages=[tokenizer, remover, word2vec])

    model = pipeline.fit(dff) # Error thrown:  Failed to execute user defined function($anonfun$createTransformFunc$1: (string) => array<string>)
    
    model.write.save(sys.argv[2])
    
    spark.stop()