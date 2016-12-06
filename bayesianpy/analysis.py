from sklearn.cross_validation import KFold
import pandas as pd
import bayesianpy.network
from bayesianpy.jni import bayesServerAnalysis
import numpy as np
import logging
from typing import Iterable, List
from collections import defaultdict

from sklearn.metrics import r2_score
def continuous_score(x, y):
    return r2_score(x, y, multioutput='uniform_average')

from sklearn.metrics import accuracy_score
def discrete_score(x, y):
    return accuracy_score(x, y['MaxStateLikelihood'])

from sklearn.metrics import confusion_matrix
def fmeasure_score(predicted, actual):
    return _fmeasure(*confusion_matrix(predicted, actual).flatten())

def _fmeasure(tp, fp, fn, tn):
    """Computes effectiveness measures given a confusion matrix."""
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)
    return { 'sensitivity': sensitivity, 'specificity': specificity, 'fmeasure': fmeasure }

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
                    name = n if names is None else n + names[i]

                    model = bayesianpy.model.NetworkModel(tpl.create(network_factory), self._logger)
                    try:
                        model.train(dataset.subset(x_train))
                    except BaseException as e:
                        self._logger.warning(e)
                        continue

                    results = model.batch_query(dataset.subset(x_test), [bayesianpy.model.QueryStatistics()], append_to_df=False)
                    ll[name].extend(results.loglikelihood.replace([np.inf, -np.inf], np.nan).tolist())

        return pd.DataFrame(ll)


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
