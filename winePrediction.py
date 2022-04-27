#prediction model 

from pyspark.mllib.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession    
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForestModel
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics
from prettytable import PrettyTable

sc = SparkContext()
spark = SparkSession(sc)
testing = spark.read.csv('s3://cs643wine/ValidationDataset.csv',header='true', inferSchema='true', sep=';')


dataFrame= testing.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
model = RandomForestModel.load(sc,"s3://cs643wine/output")

predictions = model.predict(dataFrame.map(lambda x: x.features))

labels_and_predictions = dataFrame.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(dataFrame.count())



metrics = MulticlassMetrics(labels_and_predictions)
f1 = metrics.fMeasure()
recall = metrics.recall()
precision = metrics.precision()


x = PrettyTable()
x.field_names = ["Model Accuracy", "Precision", "Recall", "F1-Score"]
x.add_row([acc * 100  ,precision, recall, f1])
print(x)