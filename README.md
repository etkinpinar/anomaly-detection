# anomaly-detection
Anomaly detection in Apache Spark and Flink with Machine Learning Libraries

## Installation

Cloning the repository

```
git clone https://github.com/etkinpinar/anomaly-detection
cd anomaly-detection
```

### Apache Spark


Running Spark container

```
cd spark
docker-compose up -d
```

Installing numpy into masternode via pip

```
docker exec masternode pip install numpy
```

### Apache Flink

Running Flink container

```
cd ../flink
docker-compose up -d
```

### User-interface

```
cd ../web-ui
pip install flask
python main.py
```
