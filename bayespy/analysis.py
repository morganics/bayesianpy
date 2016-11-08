from sklearn.cross_validation import KFold
import pandas as pd
import bayespy.network
from bayespy.jni import bayesServerAnalysis
import numpy as np
import logging
from typing import Iterable
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


class TemplateFactory:

    def __init__(self, creator_func, logger: logging.Logger, discrete=[], continuous=[]):
        self._creator_func = creator_func
        self._discrete = discrete
        self._continuous = continuous
        self._logger = logger

    def build(self, training_data: pd.DataFrame) -> bayespy.template.Template:
        return self._creator_func(training_data[self._discrete], training_data[self._continuous], self._logger)


class LogLikelihoodAnalysis:
    """
    Used for comparing models, when looking for the minimum loglikelihood value for different configurations.
    LogLikelihood cannot be used when models have a different number of variables, and can only be used between
    configurations.
    """

    def __init__(self, logger):
        self._shuffle = True
        self._logger = logger
        pass

    def analyse(self, df: pd.DataFrame, tpl_factories: Iterable[TemplateFactory], k=3):
        kf = KFold(df.shape[0], n_folds=k, shuffle=self._shuffle)
        db_folder = bayespy.utils.get_path_to_parent_dir(__file__)

        ll = defaultdict(list)
        for k, (train_indexes, test_indexes) in enumerate(kf):
            x_train, x_test = df.iloc[train_indexes], df.iloc[test_indexes]

            for i, factory in enumerate(tpl_factories):

                tpl = factory.build(df)

                with bayespy.network.NetworkFactory(x_train, db_folder, self._logger) as nf:
                    model = bayespy.model.NetworkModel(tpl.create(nf), nf.get_datastore(), self._logger)
                    model.train()

                    network = model.get_network()

                with bayespy.network.NetworkFactory(x_test, db_folder, self._logger) as nf:
                    model = bayespy.model.NetworkModel(network, nf.get_datastore(), self._logger)

                    results = model.batch_query(bayespy.model.QueryStatistics(), append_to_df=False)

                    # not ideal to remove inf, but don't want to set it to a regular number to avoid
                    # overly biasing the mean
                    ll[i].extend(results.loglikelihood.replace([np.inf, -np.inf], np.nan).tolist())

        return [np.mean(v) for k,v in ll.items()]

class RegressionAnalysis:

    def __init__(self, logger):
        self._shuffle = True
        self._logger = logger
        pass

    def analyse(self, df: pd.DataFrame, template: bayespy.template.Template, k=3):
        kf = KFold(df.shape[0], n_folds=k, shuffle=self._shuffle)
        db_folder = bayespy.utils.get_path_to_parent_dir(__file__)

        all_results = []
        for k, (train_indexes, test_indexes) in enumerate(kf):
            x_train, x_test = df.ix[train_indexes], df.ix[test_indexes]

            with bayespy.network.NetworkFactory(x_train, db_folder, self._logger) as nf:
                model = bayespy.model.NetworkModel(template.create(), nf.get_datastore(), self._logger)
                model.train()

                network = model.get_network()

            scores = []
            with bayespy.network.NetworkFactory(x_test, db_folder, self._logger) as nf:
                model = bayespy.model.NetworkModel(network, nf.get_datastore(), self._logger)
                queries = []
                for column in df.columns:
                    if bayespy.network.is_variable_continuous(bayespy.network.get_variable(network, column)):
                        queries.append(bayespy.model.QueryMeanVariance(network, column))

                results = model.batch_query(queries, append_to_df=False)
                results['k'] = k
                for column in [c for c in results.columns if c.endswith("_mean")]:
                    actual = x_test[column.replace("_mean","")]
                    predicted = x_test[column]
                    results[column.replace("_mean", "") + "_r2".format(k)] = continuous_score(actual, predicted)
                    all_results.append(results)

        return all_results
