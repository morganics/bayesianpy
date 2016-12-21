import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import bayesianpy

import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
df = pd.DataFrame({'A': [x[0] for x in diabetes_X], 'target': diabetes.target})
train, test = train_test_split(df, test_size=0.4)

logger = logging.getLogger()
bayesianpy.jni.attach(logger)
f = bayesianpy.utils.get_path_to_parent_dir(__file__)


with bayesianpy.data.DataSet(df, f, logger) as dataset:
    tpl = bayesianpy.template.NaiveBayes('target', logger, continuous=df)
    network = tpl.create(bayesianpy.network.NetworkFactory(logger))

    plt.figure()
    layout = bayesianpy.visual.NetworkLayout(network)
    graph = layout.build_graph()
    pos = layout.fruchterman_reingold_layout(graph)
    layout.visualise(graph, pos)

    model = bayesianpy.model.NetworkModel(network, logger)

    model.train(dataset.subset(train.index.tolist()))

    results = model.batch_query(dataset.subset(test.index.tolist()),
                                                            [bayesianpy.model.QueryMeanVariance('target',output_dtype=df['target'].dtype)])

    results.sort_values(by='A', ascending=True, inplace=True)
    plt.figure(figsize=(10, 10))
    plt.scatter(df['A'].tolist(), df['target'].tolist(), label='Actual')
    plt.plot(results['A'], results['target_mean'], 'ro-', label='Predicted')

    plt.fill_between(results.A, results.target_mean-results.target_variance.apply(np.sqrt),
                     results.target_mean + results.target_variance.apply(np.sqrt), color='darkgrey', alpha=0.4,
                     label='Variance'
                     )
    plt.xlabel("A")
    plt.ylabel("Predicted Target")
    plt.legend()
    plt.show()
    # plot the residuals
    plt.figure()

    plt.scatter(results['target_mean'], results['target']-results['target_mean'])
    plt.xlabel('Target')
    plt.ylabel('Target - Predicted')
    plt.show()

