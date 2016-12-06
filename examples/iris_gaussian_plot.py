import pandas as pd
import bayesianpy
from bayesianpy.network import Builder as builder

import logging
import os

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    bayesianpy.jni.attach(logger)

    db_folder = bayesianpy.utils.get_path_to_parent_dir(__file__)
    iris = pd.read_csv(os.path.join(db_folder, "data/iris.csv"), index_col=False)

    network = bayesianpy.network.create_network()
    cluster = builder.create_cluster_variable(network, 4)
    node = builder.create_multivariate_continuous_node(network, iris.drop('iris_class',axis=1).columns.tolist(), "joint")
    builder.create_link(network, cluster, node)

    class_variable = builder.create_discrete_variable(network, iris, 'iris_class', iris['iris_class'].unique())
    builder.create_link(network, cluster, class_variable)

    head_variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    with bayesianpy.data.DataSet(iris, db_folder, logger) as dataset:
        model = bayesianpy.model.NetworkModel(network, logger)
        model.train(dataset)

        queries = [bayesianpy.model.QueryConditionalJointProbability(
                head_variables=[v],
                    tail_variables=['iris_class']) for v in head_variables]

        (engine, _, _) = bayesianpy.model.InferenceEngine(network).create()
        query = bayesianpy.model.SingleQuery(network, engine, logger)
        results = query.query(queries, aslist=True)
        jd = bayesianpy.visual.JointDistribution()
        fig = plt.figure(figsize=(10,10))

        for i, r in enumerate(list(results)):
            ax = fig.add_subplot(2, 2, i+1)
            jd.plot_distribution_with_variance(ax, iris, queries[i].get_head_variables(), r)

        plt.show()

if __name__ == "__main__":
    main()