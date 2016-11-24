import pandas as pd
import bayespy
from bayespy.network import Builder as builder

import logging
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Using the latent variable to cluster data points. Based upon the Iris dataset which has 3 distinct clusters
# (not all of which are linearly separable). Using a joint probability distribution, first based upon the class
# variable 'iris_class' and subsequently the cluster variable as a tail variable. Custom query currently only supports
# a single discrete tail variable and multiple continuous head variables.

# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def main():

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    bayespy.jni.attach(logger)

    db_folder = bayespy.utils.get_path_to_parent_dir(__file__)
    iris = pd.read_csv(os.path.join(db_folder, "data/iris.csv"), index_col=False)

    network = bayespy.network.create_network()
    cluster = builder.create_cluster_variable(network, 4)
    node = builder.create_multivariate_continuous_node(network, iris.drop('iris_class',axis=1).columns.tolist(), "joint")
    builder.create_link(network, cluster, node)

    class_variable = builder.create_discrete_variable(network, iris, 'iris_class', iris['iris_class'].unique())
    builder.create_link(network, cluster, class_variable)

    def plot(head_variables, results):
        for i, hv in enumerate(head_variables):
            for j in range(i + 1, len(head_variables)):
                hv1 = head_variables[j]
                plt.figure()
                plt.plot(iris[hv].tolist(), iris[head_variables[j]].tolist(), 'bo')
                plt.title("{} vs {}".format(hv, hv1))
                for k, v in results.items():
                    mean = results[k]['mean']
                    cov = results[k]['covariance']
                    c = np.array([[cov[i,i], cov[i,j]], [cov[j,i], cov[j,j]]], np.float64)
                    plot_cov_ellipse(c, pos=(mean[i], mean[j]), nstd=3, alpha=0.5, color='green')
                plt.xlim([iris[hv].min() - 3, iris[hv].max() + 3])
                plt.ylim([iris[hv1].min() - 3, iris[hv1].max() + 3])
                plt.show()


    with bayespy.data.DataSet(iris, db_folder, logger) as dataset:
        model = bayespy.model.NetworkModel(network, logger)
        model.train(dataset)

        head_variables = ['sepal_length','sepal_width','petal_length','petal_width']

        query_type_class = bayespy.model.QueryMixtureOfGaussians(
            head_variables=head_variables,
                tail_variables=['iris_class', 'Cluster'])

        (engine, _, _) = bayespy.model.InferenceEngine(network).create()
        query = bayespy.model.SingleQuery(network, engine, logger)
        results_class = query.query([query_type_class])

        plot(head_variables, results_class)

        query_type_cluster = bayespy.model.QueryMixtureOfGaussians(
            head_variables=head_variables,
            tail_variables=['Cluster'])

        results_cluster = query.query([query_type_cluster])

        plot(head_variables, results_cluster)

if __name__ == "__main__":
    main()