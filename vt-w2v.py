from __future__ import print_function
import sys
import pyspark
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import *
# from pyspark.sql.functions import col
from pyspark.sql.types import *


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: vt-w2v.py <input> <w2v-output>", file=sys.stderr)
        exit(-1)
        
    spark = SparkSession.builder.appName('Create w2v').getOrCreate()
    
    df = spark.read.load(sys.argv[1])
    
    dff = df.filter(col('date') > '1895')
    
    # Tokenize
    # tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    # tokenized = tokenizer.transform(dff)

    # Remove stop words
    # remover = StopWordsRemover(inputCol="tokens", outputCol="words")
    # cleaned = remover.transform(tokenized)

    # Create Model
    w2v = Word2Vec(vectorSize=100, minCount=100, inputCol="text", outputCol="result")
    model = w2v.fit(dff)
    
    model.write.save(sys.argv[2])
    
    spark.stop()