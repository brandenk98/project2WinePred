#creation of model using mllib 
from pyspark.mllib.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForest

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler



sc = SparkContext()
spark = SparkSession(sc)
inputDF = spark.read.csv('s3://cs643wine/TrainingDataset.csv',header='true', inferSchema='true', sep=';')


alteredDF= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

model = RandomForest.trainClassifier(alteredDF,numClasses=10,categoricalFeaturesInfo={}, numTrees=60, maxBins=64, maxDepth=20, seed=50)
model.save(sc,"s3://cs643wine/output")







