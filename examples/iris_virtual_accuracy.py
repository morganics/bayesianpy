import pandas as pd
import bayespy
from bayespy.network import Builder as builder

import logging
import os

from bayespy.jni import jp
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def main():

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    bayespy.jni.attach(logger)

    db_folder = bayespy.utils.get_path_to_parent_dir(__file__)
    iris = pd.read_csv(os.path.join(db_folder, "data/iris.csv"), index_col=False)

    network = bayespy.network.create_network()
    num_clusters = 3
    cluster = builder.create_cluster_variable(network, num_clusters)
    node = builder.create_multivariate_continuous_node(network, iris.drop('iris_class',axis=1).columns.tolist(), "joint")
    builder.create_link(network, cluster, node)

    class_variable = builder.create_discrete_variable(network, iris, 'iris_class', iris['iris_class'].unique())
    builder.create_link(network, cluster, class_variable)

    train, test = train_test_split(iris, test_size=0.5)

    # train the model and query the most likely states and probability of each latent state.
    with bayespy.data.DataSet(iris, db_folder, logger) as dataset:
        model = bayespy.model.NetworkModel(network, logger)
        model.train(dataset.subset(train.index.tolist()))

        test_subset = dataset.subset(test.index.tolist())

        results = model.batch_query(test_subset,
                                    # creates columns Cluster$$Cluster0, Cluster$$Cluster1,
                                    # Cluster$$Cluster2, as
                                    # suffix is set to an empty string.
                                    [bayespy.model.QueryStateProbability("Cluster", suffix=""),
                                     # creates column 'iris_class_maxlikelihood'
                                     bayespy.model.QueryMostLikelyState("iris_class"),
                                     # creates column 'Cluster_maxlikelihood'
                                     bayespy.model.QueryMostLikelyState("Cluster")
                                     ])

    cluster_accuracy = {}
    # get a list of cluster accuracies, using the Bayes Server Confusion matrix class
    # weighted by the Cluster accuracy.
    with bayespy.data.DataSet(results, db_folder, logger) as resultset:
        for c in range(num_clusters):
            matrix = bayespy.jni.bayesServerAnalysis()\
                .ConfusionMatrix.create(resultset.create_data_reader_command(), "iris_class",
                                        "iris_class_maxlikelihood", "Cluster$$Cluster{}".format(c))
            cluster_accuracy.update({'Cluster{}'.format(c) : matrix.getAccuracy()})

    # generate samples from the trained model, to give us some additional testing data.
    samples = bayespy.model.Sampling(network).sample(num_samples=20).drop(["Cluster", "iris_class"], axis=1)
    reader = bayespy.data.DataFrameReader(samples)
    inference = bayespy.model.InferenceEngine(network).create_engine()
    evidence = bayespy.model.Evidence(network, inference)
    query = bayespy.model.SingleQuery(network, inference, logger)
    query_type = [bayespy.model.QueryStateProbability('Cluster', suffix="")]

    # query the expected Cluster membership, and generate a wrapper for
    # comparing the values, weighted by cluster membership.
    while reader.read():
        result = query.query(query_type, evidence=evidence.apply(reader.to_dict()))
        cv_results = []
        for i, (key,value) in enumerate(result.items()):
            n = bayespy.network.Discrete.fromstring(key)
            weighting = cluster_accuracy[n.state]
            cv_results.append(bayespy.jni.bayesServerAnalysis().DefaultCrossValidationTestResult(
                jp.JDouble(weighting), jp.JObject(value, jp.java.lang.Object), jp.java.lang.Double(jp.JDouble(value)))
            )


        score = bayespy.jni.bayesServerAnalysis().CrossValidation.combine(jp.java.util.Arrays.asList(cv_results),
                                    bayespy.jni.bayesServerAnalysis().CrossValidationCombineMethod.WEIGHTED_AVERAGE)

        # append the score on to the existing dataframe
        samples.set_value(reader.get_index(), 'score', score)

    variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    cmap = plt.cm.get_cmap('Blues')
    fig = plt.figure(figsize=(10, 10))
    k = 1
    # plot!
    for i, v in enumerate(variables):
        for j in range(i + 1, len(variables)):
            v1 = variables[j]
            ax = fig.add_subplot(3, 2, k)
            ax.set_title("{} vs {}".format(v, v1))
            ax.scatter(x=iris[v].tolist(), y=iris[v1].tolist(), facecolors='none', alpha=0.1)
            h = ax.scatter(x=samples[v].tolist(), y=samples[v1].tolist(), c=samples['score'].tolist(),
                           vmin=samples.score.min(), vmax=samples.score.max(), cmap=cmap
                           )
            k += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(h, cax=cbar_ax)
    plt.show()

if __name__ == "__main__":
    main()