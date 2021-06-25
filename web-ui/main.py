from flask import Flask, render_template, request
import os

results = []

app = Flask(__name__)


@app.route('/')
def post_result():
    return render_template("index.html")


@sio.event
def connect():
    print("I'm connected!")


@sio.event
def connect_error(data):
    print("The connection failed!")


@app.route('/train', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':

        framework = request.form.get("framework")
        algorithm = request.form.get("algorithm")
        multi_node = request.form.get("multi-node")
        ratio = float(request.form.get("ratio")) / 100
        input_file = request.form.get('input-file')

        if multi_node != 1:
            multi_node = 1
        else:
            multi_node = 2

        # command = "docker exec masternode ./bin/spark-submit --total-executor-cores 2 anomaly.py --input data-anomaly-dropped.csv --algorithm svm"

        if framework == "spark":
            command = "docker exec " \
                      "masternode ./bin/spark-submit " \
                      "--total-executor-cores {} " \
                      "anomaly.py " \
                      "--input {} " \
                      "--ratio {} " \
                      "--algorithm {}".format(multi_node, input_file, ratio, algorithm)
        else:
            command = "docker exec jobmanager ./bin/flink run -c anomaly.Main anomaly.jar " \
                      "--input {} --paralellism {} --ratio {}".format(input_file, multi_node, ratio)

        stream = os.popen(command)
        output = stream.read()
        splitted = output.split('<->')

        return render_template("index.html",
                               accuracy=splitted[1],
                               f1_score=splitted[2],
                               precision=splitted[3],
                               sensitivity=splitted[4],
                               trainTime=splitted[6],
                               testTime=splitted[8])


app.run()

