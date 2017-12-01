from sklearn.model_selection import KFold as NewKFold, StratifiedKFold
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
    return {'sensitivity': sensitivity, 'specificity': specificity, 'precision': tp / (tp + fp),
            'fmeasure': fmeasure,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'positive_likelihood_ratio': sensitivity / (1 - specificity),
            'negative_likelihood_ratio': (1 - sensitivity) / specificity}


def predictive_value(predicted, actual, labels=None):
    def _predictive_value(tp, fp, fn, tn):
        return {'positive_predictive_value': tp / (tp + fp),
                'negative_predictive_value': tn / (tn + fn)}

    return _predictive_value(*confusion_matrix(predicted, actual, labels=labels).flatten())


from bayesianpy.network import Builder as builder


class DiscretisationAnalysis:
    def __init__(self, logger, shuffle=True):
        self._shuffle = shuffle
        self._logger = logger

    def analyse(self, df: pd.DataFrame, continuous_variable_names: List[str]):
        kf = NewKFold(n_splits=3, shuffle=self._shuffle)

        network_factory = bayesianpy.network.NetworkFactory(self._logger)
        variations = [1, 5, 10, 20, 30]
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

class Serialiser:
    def __init__(self):
        pass

    def save(self, model, filename):
        pass

class FileSerialiser(Serialiser):
    def __init__(self, filepath):
        self._filepath = filepath

    def save(self, model, filename):
        if not os.path.exists(self._filepath):
            os.mkdir(self._filepath)

        if isinstance(model, bayesianpy.model.NetworkModel):
            model = model.get_network()

        bayesianpy.network.save(model, os.path.join(self._filepath, filename))

class CrossValidatedAnalysis:
    """
        Used for comparing models
        """

    def __init__(self, logger:logging.Logger, shuffle:bool=True, serialiser:Serialiser=Serialiser()):
        self._shuffle = shuffle
        self._logger = logger
        self._models = []
        self._cv_method = None
        self._serialiser = serialiser

    def get_models(self) -> List[bayesianpy.model.NetworkModel]:
        return self._models

    def _get_cv_splits(self, df) -> Iterable:
        pass

    def analyse(self, df: pd.DataFrame, templates: List[bayesianpy.template.Template], dataset, queries,
                append_to_df=True, maximum_iterations=100, include_model=False):
        network_factory = bayesianpy.network.NetworkFactory(self._logger)

        for k, (x_train, x_test) in enumerate(self._get_cv_splits(df)):
            for tpl in templates:
                nt = tpl.create(network_factory)
                nt = bayesianpy.network.remove_single_state_nodes(nt)
                model = bayesianpy.model.NetworkModel(nt, self._logger)

                if self._serialiser is not None:
                    self._serialiser.save(model, "pretrained.{}.bayes".format(k))

                self._models.append(model)

                try:
                    model.train(dataset.subset(x_train.index), maximum_iterations=maximum_iterations)
                    if self._serialiser is not None:
                        self._serialiser.save(model, "trained.{}.bayes".format(k))

                except BaseException as e:
                    self._logger.warning(e)
                    continue


                result = model.batch_query(dataset.subset(x_test.index), queries,
                                        append_to_df=append_to_df)

                if include_model:
                    yield model, result
                else:
                    yield result


class KFoldAnalysis(CrossValidatedAnalysis):
    """
    Used for comparing models
    """

    def __init__(self, logger, shuffle=True, kfolds=3):
        super().__init__(logger, shuffle)
        self._kfolds = kfolds

    def _get_cv_splits(self, df):
        if self._cv_method is None:
            self._cv_method = NewKFold(n_splits=self._kfolds, shuffle=self._shuffle)

        for train, test in self._cv_method.split(df):
            yield df.ix[train], df.ix[test]


class StratifiedKFoldAnalysis(CrossValidatedAnalysis):
    """
    Used for comparing models
    """

    def __init__(self, target_col:str, logger, shuffle=True, kfolds=3, serialiser=None):
        super().__init__(logger, shuffle, serialiser=serialiser)
        self._kfolds = kfolds
        self._class_col = target_col
        self._serialiser = serialiser

    def _get_cv_splits(self, df):
        if self._cv_method is None:
            self._cv_method = StratifiedKFold(n_splits=self._kfolds, shuffle=self._shuffle)

        for train, test in self._cv_method.split(df, df[self._class_col]):
            yield df.ix[train], df.ix[test]

from typing import Callable
class CustomAnalysis(CrossValidatedAnalysis):
    """
    Used for comparing models
    """

    def __init__(self, logger,
                 train_selector:Callable[[pd.DataFrame], pd.DataFrame],
                 test_selector: Callable[[pd.DataFrame], pd.DataFrame]):
        super().__init__(logger, shuffle=False)
        self._train_selector = train_selector
        self._test_selector = test_selector

    def _split(self, df):
        return self._train_selector(df), self._test_selector(df)

    def _get_cv_splits(self, df):
        return [self._split(df)]



class TrainTestSplitAnalysis(CrossValidatedAnalysis):
    """
    Used for comparing models
    """

    def __init__(self, logger, shuffle=True, kfolds=3):
        super().__init__(logger, shuffle)
        self._kfolds = kfolds

    def _get_cv_splits(self, df):
        self._logger.info("Not KFold anymore, just doing a train/test split.")
        x_train, x_test = train_test_split(df, test_size=0.33)
        return [(x_train, x_test)]

class DummyAnalysis(CrossValidatedAnalysis):
    """
    Used for comparing models
    """

    def __init__(self, logger):
        super().__init__(logger, False)

    def _get_cv_splits(self, df):
        self._logger.info("Testing and training on the same data")
        return [(df, df)]


class LogLikelihoodAnalysis:
    """
    Used for comparing models, when looking for the minimum loglikelihood value for different configurations.
    LogLikelihood cannot be used when models have a different number of variables, and can only be used between
    configurations.
    """

    def __init__(self, logger, split_strategy:CrossValidatedAnalysis):
        self._logger = logger
        self._split_strategy = split_strategy

    def analyse(self, df: pd.DataFrame, templates: List[bayesianpy.template.Template], dataset):

        ll = defaultdict(list)

        for tpl in templates:
            label = tpl.get_label()
            for r in self._split_strategy.analyse(df, [tpl], dataset, [bayesianpy.model.QueryModelStatistics()], append_to_df=False):
                ll[label].extend(r.loglikelihood.replace([np.inf, -np.inf], np.nan).tolist())

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
