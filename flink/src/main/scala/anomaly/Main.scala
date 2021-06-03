package anomaly

/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.{LabeledVector, ParameterMap, WeightVector}
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.classification.SVM
import org.apache.flink.ml.optimization.{GenericLossFunction, GradientDescentL1, LearningRateMethod, LinearPrediction, SquaredLoss}
import org.apache.flink.ml.preprocessing.{MinMaxScaler, Splitter}
import org.apache.flink.ml.preprocessing.Splitter.TrainTestDataSet
import org.apache.flink.ml.recommendation.ALS
import org.apache.flink.ml.regression.MultipleLinearRegression

import java.util.concurrent.TimeUnit

object Main {
  def createConfMtx( evaluationPairs: DataSet[(Double, Double)]): Array[Int] = {
    var conf_mtx = Array(0,0,0,0)

    val dataSetList = evaluationPairs.collect()
    for (e <- dataSetList) {
      if (e._1 == e._2)
          conf_mtx(0) += 1
      else {
        if (e._1 == 0) {
          if ( e._2 == -1)
            conf_mtx(3) += 1
          else
            conf_mtx(2) += 1
        } else
          conf_mtx(1) += 1
      }
    }
  return conf_mtx
  }
  def main(args: Array[String]) {
    val params: ParameterTool = ParameterTool.fromArgs(args)

    // set up the execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment

    // set up parameters
    val parallelism =
      if (params.has("parallelism"))
        params.getInt("parallelism")
      else {
        println("Parallelism is set to 1 as default value.")
        1
      }
    env.setParallelism(parallelism)

    val algorithm_choice =
      if (params.has("algorithm"))
        params.get("algorithm")
      else {
        println("Algorithm is set to SVM as default value.")
        "SVM"
      }

    val data =
      if (params.has("input")) {
        env.readCsvFile[(String, String, String, String, String, String, String)](
          params.get("input"))
      } else {
        println("Data file is not provided. Shutting down..")
        return
      }

    // prepare data for ML algorithms
    val dataLV = data
      .map{tuple =>
        val list = tuple.productIterator.toList
        val numList = list.map(_.asInstanceOf[String].toDouble)
        LabeledVector(numList(6), DenseVector(numList.take(6).toArray))
      }

    val trainTestRatio = 0.8
    val dataTrainTest: TrainTestDataSet[LabeledVector] = Splitter.trainTestSplit(dataLV, trainTestRatio, true)

    var evaluation: DataSet[(Double, Double)] = null

    if (algorithm_choice.equalsIgnoreCase("SVM")){
      val svm = SVM()
        .setBlocks(env.getParallelism)
        .setIterations(150)
        .setRegularization(0.01)
        .setStepsize(0.1)
        .setThreshold(1)

      svm.fit(dataTrainTest.training)

      evaluation = svm.evaluate(dataTrainTest.testing.map(x => (x.vector, x.label)))
    }
    else if (algorithm_choice.equalsIgnoreCase("MLR")){
      val mlr = MultipleLinearRegression()
        .setIterations(100)
        .setStepsize(0.1)
        .setConvergenceThreshold(0.001)

      mlr.fit(dataTrainTest.training)

      evaluation = mlr.evaluate(dataTrainTest.testing.map( x => (x.vector, x.label)))
    }
    else if (algorithm_choice.equalsIgnoreCase("DT")){
      /*
      val dt = DecisionTree()
        .setClasses(2)
        .setMaxBins(3)
        .setDepth(10)
        .setDimension(6)

      dt.fit(dataTrainTest.training)

      evaluation = dt.evaluate(dataTrainTest.testing.map( x => (x.vector, x.label)))
      evaluation.print()
      */
    }
    else {
      println("Choice of algorithm is not valid. Program shutting down..")
      return
    }
    println(algorithm_choice.toUpperCase() + " Training Time: " + env.getLastJobExecutionResult.getNetRuntime(TimeUnit.MILLISECONDS) + " ms")

    var sampleSize = 0
    val conf_mtx = createConfMtx(evaluation)

    println("--- Confusion Matrix ---")
    for ( i <- conf_mtx.indices) {
      if ( i == 2)
        println()
      print(conf_mtx(i) + " ")
      sampleSize += conf_mtx(i)
    }
    println("\n--- Confusion Matrix ---\n")

    println(algorithm_choice.toUpperCase() + " Accuracy: " + (conf_mtx(0) + conf_mtx(3)) / sampleSize.toFloat)
    println(algorithm_choice.toUpperCase() + " F1 Score: " + 2*conf_mtx(0) / (2*conf_mtx(0) + conf_mtx(1) + conf_mtx(2)).toFloat)
    println(algorithm_choice.toUpperCase() + " Sensitivity: " + conf_mtx(0) / (conf_mtx(0) + conf_mtx(2)).toFloat)
    println(algorithm_choice.toUpperCase() + " Precision: " + conf_mtx(0) / (conf_mtx(0) + conf_mtx(1)).toFloat)

    /*
    val sgd = GradientDescentL1()
      .setLossFunction(GenericLossFunction(SquaredLoss, LinearPrediction))
      .setRegularizationConstant(0.2)
      .setIterations(100)
      .setConvergenceThreshold(0.001)

    val initWeights: DataSet[WeightVector] = sgd.createInitialWeightVector(env.fromElements(6))
    // Optimize the weights, according to the provided data
    val weightDS: DataSet[WeightVector] = sgd.optimize(dataTrainTest.training, Some(initWeights))
    weightDS.print()
    */

  }
}
