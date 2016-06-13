from __future__ import print_function
# PySpark
import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row, HiveContext
#from pyspark.sql import HiveContext, Row
from pyspark.sql.types import DataType, IntegerType
# mllib for clustering
from pyspark.mllib.linalg import Vectors, DenseMatrix
from pyspark.mllib.clustering import GaussianMixture, KMeansModel, KMeans
# JSON
import json
import collections
# numpy
from numpy.testing import assert_equal
import numpy as np
from shutil import rmtree
from numpy import array
from datetime import timedelta, date

import random


if __name__ == "__main__":
    sqlsc = SQLContext(sc)
    MYSQL_USERNAME = ""
    MYSQL_PWD = ""
    #Original URL
    MYSQL_CONNECTION_URL = "jdbc:mysql://localhost:3306/telegramdb?autoReconnect=true&useSSL=false&user=" + MYSQL_USERNAME + "&password=" + MYSQL_PWD

    info_df = sqlsc.read.format("jdbc").options(
        url = MYSQL_CONNECTION_URL,
        dbtable = "information",
        driver = "com.mysql.jdbc.Driver"
    ).load()
    tag_df = sqlsc.read.format("jdbc").options(
        url = MYSQL_CONNECTION_URL,
        dbtable = "tags",
        driver = "com.mysql.jdbc.Driver"
    ).load()
    col_num = tag_df.filter(tag_df.high == 'IT').count()
    tags = tag_df.filter(tag_df.high == 'IT').map(lambda list: list.low).collect()
    cols = {}
    for tag in tags:
        cols[tag]=0
        #print(tag)
    print(cols)
    #results = info.map(lambda line: array([x[1:-1].replace("{", "").replace("}","") for x in line.low.split(",")])).collect()
    #for temp in results:
    #    print(temp)
    repos = info_df.filter(info_df.high == 'IT').map(lambda line: {line.pk_aid:json.loads(line.low,
                                        encoding="utf-8")}).collect()
    rows = info_df.filter(info_df.high == 'IT').map(lambda line: {line.pk_aid:np.zeros(col_num, dtype=np.int)}).collect()
    row_num = info_df.filter(info_df.high == 'IT').count()
    #print(row_num)
    print(row_num)
    print(col_num)
    #print(data)
    #print(rows)
    for index, repo in enumerate(repos):
        #print("[%d] : "%(index)+str(temps))
        for temp in repo:
            print("[%d] : "%(index)+str(repo.get(temp)))
            for element in repo.get(temp):
                t = element.items()
                print("->" + str(t) + ", ")
                      #+ str(element.get(element)))
        #for temp in cols:
            #print("[%d] : "%(index)+str(temp))
    for index, repo in enumerate(repos):
        for pk_aids in repo:
            elements = repo.get(pk_aids)
            for element in elements:
                for col_index, col in enumerate(cols):
                    if element.get(col) is not None:
                        rows[index].get(pk_aids)[col_index]=element.get(col)*2
                        #print(element.get(col))
                    else:
                        rows[index].get(pk_aids)[col_index]=random.randrange(2)
    for index, row in enumerate(rows):
        for pk_aids in row:
            if rows[index].get(pk_aids) is not None:
                #print(rows[index].get(pk_aids))
                if index == 0:
                    data = rows[index].get(pk_aids)
                else:
                    data = np.append(data, rows[index].get(pk_aids))

    print(str(np.resize(data, (row_num, col_num)).shape))
    #data = np.resize(data, (row_num, col_num)
    clusterdata_1 = sc.parallelize(np.resize(data, (row_num, col_num)))
    model = KMeans.train(clusterdata_1, 10, maxIterations=100, runs=30,
                         initializationMode="random", seed=10, initializationSteps=10, epsilon=1e-4)
    #model = GaussianMixture.train(clusterdata_1, 3, convergenceTol=0.9, maxIterations=100, seed=10)
    #for i in range(3):
    #    print ("weight = ", model.weights[i], "mu = ", model.gaussians[i].mu,
    #        "sigma = ", model.gaussians[i].sigma.toArray())
    labels = model.predict(clusterdata_1).collect()
    print(labels)
