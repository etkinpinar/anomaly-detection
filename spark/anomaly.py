
import time, argparse
from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, NaiveBayes, RandomForestClassifier, LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', help='Input csv file', default= None)
    parser.add_argument('--algorithm', help='Choice of algorithm -> SVM - DT (Decision Tree) - NB (Naive Bayes) - RF (Random Forest) - LR (Logistic Regression) | Default: SVM', default= "svm")
    parser.add_argument('--ratio', help='Train ratio | Default: 0.8', default= 0.8)

    args = parser.parse_args()

    spark = SparkSession.builder.appName("Anomaly Detection Application").getOrCreate()

    # read input file
    try:
        data = spark.read.format("csv").option("inferSchema", "true").load(args.input)
    except:
        print("Input file is not valid.")
        exit()

    # under sample the majority label
    normal = data.where("{} == 0".format(str(data.columns[-1])))
    anomaly = data.where("{} == 1".format(str(data.columns[-1])))
    data = normal.sample(float(anomaly.count()) / float(normal.count()))
    data = data.union(anomaly)

    # prepare data to train models
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
    data = data.withColumnRenamed(data.columns[-1],"label")
    data = assembler.transform(data)
    for col in data.columns[:-2]:
        data = data.drop(col)
    
    # train-test data split
    trainRatio = float(args.ratio)
    trainingData, testData = data.randomSplit([trainRatio, 1-trainRatio])

    algorithm_choice = args.algorithm
    
    t0 = time.time() 
    if algorithm_choice.lower() == "svm":
        svm = LinearSVC(maxIter=100, regParam=0.001, threshold=0.7)   
        model = svm.fit(trainingData)

    elif algorithm_choice.lower() == "dt":
        dt = DecisionTreeClassifier()
        model = dt.fit(trainingData)
    
    elif algorithm_choice.lower() == "nb":
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
        model = nb.fit(trainingData)

    elif algorithm_choice.lower() == "rf":
        rf = RandomForestClassifier(numTrees=10)
        rf = LogisticRegression(maxIter=50, elasticNetParam=0.8, family="multinomial")
        model = rf.fit(trainingData)

    elif algorithm_choice.lower() == "lr":
        lr = LogisticRegression(maxIter=50, elasticNetParam=0.8, family="multinomial")
        model = lr.fit(trainingData)

    else:
        print("Choice of algorithm is not valid. Program shutting down..")
        exit(-1)   
    timePassed = (time.time() - t0) * 1000 

    # prediction phase
    predictions = model.transform(testData)
    
    # get result metrics
    acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(predictions)
    prec = MulticlassClassificationEvaluator(metricName="precisionByLabel", metricLabel= 1).evaluate(predictions)
    sens = MulticlassClassificationEvaluator(metricName="recallByLabel", metricLabel= 1).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator().evaluate(predictions)

    # print results
    print("<->{:.6f}".format(acc), end='<->')
    print("{:.6f}".format(f1), end='<->')
    print("{:.6f}".format(prec), end='<->')
    print("{:.6f}".format(sens), end='<->')
    print("{:.3f}".format(timePassed), end='<->')

    spark.stop()

# run it in docker
# docker exec -it masternode ./bin/spark-submit --total-executor-cores 1 anomaly.py --input data-anomaly-dropped.csv --algorithm dt