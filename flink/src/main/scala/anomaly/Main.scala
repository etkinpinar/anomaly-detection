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
import org.apache.flink.ml.classification.SVM
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.preprocessing.Splitter
import org.apache.flink.ml.preprocessing.Splitter.TrainTestDataSet

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

    val trainRatio: Double =
      if (params.has("ratio"))
        params.get("ratio").toDouble
      else {
        println("Train ratio set to 0.8 as default value.")
        0.8
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
    var dataLV = data
      .map{tuple =>
        val list = tuple.productIterator.toList
        val numList = list.map(_.asInstanceOf[String].toDouble)
        LabeledVector(numList(6), DenseVector(numList.take(6).toArray))
      }

    // under sample the majority label
    val anomalyDS = dataLV.filter(_.label == 1)
    val normalDS = dataLV.filter(_.label == 0)
    val anomalyCount = anomalyDS.count()
    val normalCount = normalDS.count()
    val sampledNormal = Splitter.randomSplit(normalDS, anomalyCount.toDouble / normalCount, precise = true)
    dataLV = anomalyDS.union(sampledNormal(0))

    val dataTrainTest: TrainTestDataSet[LabeledVector] = Splitter.trainTestSplit(dataLV, trainRatio, true)

    val svm = SVM()
      .setBlocks(env.getParallelism)
      .setIterations(100)
      .setRegularization(0.001)
      .setStepsize(0.1)
      .setThreshold(0.7)

    val trainT0 = System.nanoTime
    svm.fit(dataTrainTest.training)
    val trainTime: DataSet[String] = env.fromElements("<->" + ((System.nanoTime - trainT0) / 1e6d).toString + "<->")

    val evalT0 = System.nanoTime
    val evaluation = svm.evaluate(dataTrainTest.testing.map(x => (x.vector, x.label)))
    val evalTime: DataSet[String] = env.fromElements("<->" + ((System.nanoTime - evalT0) / 1e6d).toString + "<->")

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

    print("<->" + (conf_mtx(0) + conf_mtx(3)) / sampleSize.toFloat)
    print("<->" + 2*conf_mtx(0) / (2*conf_mtx(0) + conf_mtx(1) + conf_mtx(2)).toFloat)
    print("<->" + conf_mtx(0) / (conf_mtx(0) + conf_mtx(1)).toFloat)
    print("<->" + conf_mtx(0) / (conf_mtx(0) + conf_mtx(2)).toFloat + "<->")
    //print("<->" + env.getLastJobExecutionResult.getNetRuntime(TimeUnit.MILLISECONDS) + "<->")
    trainTime.print()
    evalTime.print()
  }
}
