from sklearn.cross_validation import KFold
import pandas as pd
import bayesianpy.network
from bayesianpy.jni import bayesServerAnalysis
import numpy as np
import logging
from typing import Iterable, List
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
def continuous_score(x, y):
    return r2_score(x, y, multioutput='uniform_average')

from sklearn.metrics import accuracy_score
def discrete_score(x, y):
    return accuracy_score(x, y['MaxStateLikelihood'])

from sklearn.metrics import confusion_matrix
def fmeasure_score(predicted, actual, labels=None):
    return _fmeasure(*confusion_matrix(actual, predicted, labels=labels).flatten())

def _fmeasure(tp, fp, fn, tn):
    """Computes effectiveness measures given a confusion matrix."""
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)
    return { 'sensitivity': sensitivity, 'specificity': specificity, 'precision': tp / (tp + fp),
        'fmeasure': fmeasure,
             'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
             'positive_likelihood_ratio': sensitivity / (1 - specificity),
             'negative_likelihood_ratio': (1- sensitivity) / specificity}

def predictive_value(predicted, actual, labels=None):
    def _predictive_value(tp, fp, fn, tn):
        return {'positive_predictive_value': tp / (tp + fp),
            'negative_predictive_value': tn/(tn + fn)}

    return _predictive_value(*confusion_matrix(predicted, actual, labels=labels).flatten())

from bayesianpy.network import Builder as builder

class DiscretisationAnalysis:

    def __init__(self, logger, shuffle=True):
        self._shuffle = shuffle
        self._logger = logger

    def analyse(self, df: pd.DataFrame, continuous_variable_names:List[str]):
        kf = KFold(df.shape[0], n_folds=3, shuffle=self._shuffle)

        network_factory = bayesianpy.network.NetworkFactory(self._logger)
        variations = [1,5,10,20,30]
        results = {}
        with bayesianpy.data.DataSet(df, logger=self._logger) as dataset:
            ll = defaultdict(list)
            for variable in continuous_variable_names:
                likelihoods = []
                for cluster_count in variations:
                    weighted = []
                    weights = []
                    for k, (train_indexes, test_indexes) in enumerate(kf):


                        x_train, x_test = train_indexes, test_indexes

                        nt = network_factory.create()
                        cluster = builder.create_cluster_variable(nt, cluster_count)
                        node = builder.create_continuous_variable(nt, variable)
                        builder.create_link(nt, cluster, node)

                        model = bayesianpy.model.NetworkModel(nt, self._logger)

                        try:
                            ll = model.train(dataset.subset(x_train)).get_metrics()['loglikelihood']
                        except BaseException as e:
                            self._logger.warning(e)
                            continue

                        weighted.append(ll)
                        weights.append(len(x_train))

                    likelihoods.append(np.average(weighted, weights=weights))

                max_index = np.argmax(likelihoods)
                if variations[max_index] > 5:
                    results.update({variable: True})
                else:
                    results.update({variable: False})

        return results

class LogLikelihoodAnalysis:
    """
    Used for comparing models, when looking for the minimum loglikelihood value for different configurations.
    LogLikelihood cannot be used when models have a different number of variables, and can only be used between
    configurations.
    """

    def __init__(self, logger, shuffle=True):
        self._shuffle = shuffle
        self._logger = logger

    def analyse(self, df: pd.DataFrame, templates: Iterable[bayesianpy.template.Template], k=3, names: List[str] = None,
                use_model_names=True):
        kf = KFold(df.shape[0], n_folds=k, shuffle=self._shuffle)
        db_folder = bayesianpy.utils.get_path_to_parent_dir(__file__)

        network_factory = bayesianpy.network.NetworkFactory(self._logger)
        with bayesianpy.data.DataSet(df, db_folder, self._logger) as dataset:
            ll = defaultdict(list)
            for k, (train_indexes, test_indexes) in enumerate(kf):
                x_train, x_test = train_indexes, test_indexes

                for i, tpl in enumerate(templates):

                    n = type(tpl).__name__ if use_model_names else ""
                    name = n if names is None else n + str(names[i])

                    model = bayesianpy.model.NetworkModel(tpl.create(network_factory), self._logger)
                    try:
                        model.train(dataset.subset(x_train))
                    except BaseException as e:
                        self._logger.warning(e)
                        continue

                    results = model.batch_query(dataset.subset(x_test), [bayesianpy.model.QueryStatistics()], append_to_df=False)
                    ll[name].extend(results.loglikelihood.replace([np.inf, -np.inf], np.nan).tolist())

        return pd.DataFrame(ll)

class KFoldAnalysis:
    """
    Used for comparing models
    """

    def __init__(self, logger, shuffle=True):
        self._shuffle = shuffle
        self._logger = logger
        self._models = []

    def get_models(self) -> List[bayesianpy.model.NetworkModel]:
        return self._models

    def analyse(self, df: pd.DataFrame, tpl: bayesianpy.template.Template, dataset, queries, k=3, append_to_df=True):
        if k > 1:
            kf = KFold(df.shape[0], n_folds=k, shuffle=self._shuffle)
        else:
            self._logger.info("Not KFold anymore, just doing a train/test split.")
            x_train, x_test = train_test_split(df, test_size=0.33)
            kf = [(x_train.index, x_test.index)]

        network_factory = bayesianpy.network.NetworkFactory(self._logger)

        for k, (x_train, x_test) in enumerate(kf):

            self._logger.info("Running KFold {} of {}".format(k, len(kf)))
            model = bayesianpy.model.NetworkModel(tpl.create(network_factory), self._logger)
            self._models.append(model)
            try:
                model.train(dataset.subset(x_train))
            except BaseException as e:
                self._logger.warning(e)
                continue

            model.save("trained.bayes")
            yield model.batch_query(dataset.subset(x_test), queries,
                                      append_to_df=append_to_df)

# class RegressionAnalysis:
#
#     def __init__(self, logger):
#         self._shuffle = True
#         self._logger = logger
#         pass
#
#     def analyse(self, df: pd.DataFrame, template: bayespy.template.Template, k=3):
#         kf = KFold(df.shape[0], n_folds=k, shuffle=self._shuffle)
#         db_folder = bayespy.utils.get_path_to_parent_dir(__file__)
#
#         all_results = []
#         for k, (train_indexes, test_indexes) in enumerate(kf):
#             x_train, x_test = df.ix[train_indexes], df.ix[test_indexes]
#
#             with bayespy.network.NetworkFactory(x_train, db_folder, self._logger) as nf:
#                 model = bayespy.model.NetworkModel(template.create(nf), nf.get_datastore(), self._logger)
#                 model.train()
#
#                 network = model.get_network()
#
#             scores = []
#             with bayespy.network.NetworkFactory(x_test, db_folder, self._logger) as nf:
#                 model = bayespy.model.NetworkModel(network, nf.get_datastore(), self._logger)
#                 queries = []
#                 for column in df.columns:
#                     if bayespy.network.is_variable_continuous(bayespy.network.get_variable(network, column)):
#                         queries.append(bayespy.model.QueryMeanVariance(network, column))
#
#                 results = model.batch_query(queries, append_to_df=False)
#                 results['k'] = k
#                 for column in [c for c in results.columns if c.endswith("_mean")]:
#                     actual = x_test[column.replace("_mean","")]
#                     predicted = x_test[column]
#                     results[column.replace("_mean", "") + "_r2".format(k)] = continuous_score(actual, predicted)
#                     all_results.append(results)
#
#         return all_results
