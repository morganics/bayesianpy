from bayespy.network import NetworkFactory

class Selector:
    def __init__(self, target, continuous=[], discrete=[]):
        self.target = target
        self._continuous = list(continuous)
        self._discrete = list(discrete)
        self._index = -1
        self._all_variables = self._continuous + self._discrete
        self._tix = self._all_variables.index(target)

    def _c_length(self):
        return len(self._continuous)

    def _d_length(self):
        return len(self._discrete)

    def _is_continuous(self, ix):
        return ix < self._c_length()

    def _is_discrete(self, ix):
        return ix >= self._c_length()

    def next_combination(self):
        self._index += 1

        if self._index >= len(self._all_variables):
            return False

        return True

class UnivariateSelector(Selector):

    def __init__(self, target, continuous=[], discrete=[]):
        Selector.__init__(self, target, continuous, discrete)

    def _get_vars(self):
        return [self._all_variables[self._index], self._all_variables[self._tix]]

    def get_discrete_variables(self):
        for v in self._get_vars():
            if v in self._discrete:
                yield v

    def get_continuous_variables(self):
         for v in self._get_vars():
            if v in self._continuous:
                yield v

    def get_key_variables(self):
        variables = self._get_vars()

        variables.pop(variables.index(self.target))
        return variables

    def next_combination(self):
        self._index += 1

        if self._index >= len(self._all_variables):
            return False

        v = self._get_vars()
        if len(np.unique(v)) < len(v):
            return self.next_combination()

        return True

import numpy as np
import itertools
class CartesianProductSelector(Selector):
    def __init__(self, target, continuous=[], discrete=[], n=2):

        Selector.__init__(self, target, continuous, discrete)
        self._total = len(self._all_variables)

        if n > self._total - 1:
            raise ValueError("n needs to be less or equal to the total length of the variable array - 1")

        list_of_lists = [range(self._total) for _ in range(n)]
        self._combinations = list(set(itertools.product(*list_of_lists)))
        self._tix = self._all_variables.index(target)

    def _get_vars(self):
        r=[]

        for ix in self._indexes:
            r.append(self._all_variables[ix])

        if self.target not in r:
            r.append(self.target)

        return r

    def get_total_combinations(self):
        return len([item for item in self._combinations if self._filter_combination(item)])

    def _filter_combination(self, combination):
        if len(np.unique(combination)) != len(combination):
            return False

        if self._tix in combination:
            return False

        return True

    def get_key_variables(self):
        variables = self._get_vars()
        variables.pop(variables.index(self.target))
        return variables

    def get_discrete_variables(self):

        for v in self._get_vars():
            if v in self._discrete:
                yield v

    def get_continuous_variables(self):

        for v in self._get_vars():
            if v in self._continuous:
                yield v

    def next_combination(self):
        self._index += 1

        if self._index >= len(self._combinations):
            return False

        self._indexes = list(self._combinations[self._index])

        if not self._filter_combination(self._indexes):
            return self.next_combination()

        return True

#l = CartesianProductSelector('c1', continuous=['c1', 'c2', 'c3'], discrete=['d1','d2','d3'], n=4)
#from collections import Counter
#c = Counter()
#while l.next_combination():
#
#    for i in [v for v in l.get_continuous_variables()]:
#        c[i] += 1
#    for i in [v for v in l.get_discrete_variables()]:
#        c[i] += 1

class LeaveSomeOutSelector(Selector):

    def __init__(self, target, continuous=[], discrete=[], some=1):
        Selector.__init__(self, target, continuous, discrete)
        self._some = some
        if some > len(self._all_variables):
            raise ValueError("Some cannot be greater than the total number of columns")

    def _get_vars(self):
        variables = list(self._all_variables)
        start_index = self._index
        if self._index + self._some > len(variables):
            start_index = abs(len(variables) - (self._index + self._some))
            r = variables[start_index : self._index]
        else:
            r = variables[ : start_index] + variables[start_index + self._some:]

        if self.target not in r:
            r.append(self.target)

        return r

    def get_discrete_variables(self):
        if self._index == -1:
            raise ValueError("Call next_combination first")
        for v in self._get_vars():
            if v in self._discrete:
                yield v

    def get_continuous_variables(self):
        if self._index == -1:
            raise ValueError("Call next_combination first")
        for v in self._get_vars():
            if v in self._continuous:
                yield v

    def get_key_variables(self):
        if self._index == -1:
            raise ValueError("Call next_combination first")

        return list(set(self._all_variables) - set(self._get_vars()))


#l = LeaveSomeOutSelector('c1', continuous=['c1', 'c2', 'c3'], discrete=['d1', 'd2', 'd3'], some=3)
#while l.next_combination():
#    print(l._index)
#    print(list(l.get_continuous_variables()))
#    print(list(l.get_discrete_variables()))

class IterativeSelector(Selector):

    def __init__(self, target, continuous=[], discrete=[], ordering=[], addition=True):
        Selector.__init__(self, target, continuous, discrete)
        self._ordering = ordering
        self._addition = addition

    def _get_vars(self):
        variables = list(self._ordering)
        start_index = self._index + 1

        if self._addition:
            r = variables[:start_index]
        else:
            r = variables[-start_index:]

        if self.target not in r:
            r.append(self.target)

        return r

    def get_discrete_variables(self):
        if self._index == -1:
            raise ValueError("Call next_combination first")

        for v in self._get_vars():
            if v in self._discrete:
                yield v

    def get_continuous_variables(self):
        if self._index == -1:
            raise ValueError("Call next_combination first")

        for v in self._get_vars():
            if v in self._continuous:
                yield v

    def get_key_variables(self):
        if self._index == -1:
            raise ValueError("Call next_combination first")

        variables = self._get_vars()
        variables.pop(variables.index(self.target))
        return variables

    def next_combination(self):
        self._index += 1

        if self._index >= len(self._ordering):
            return False

        return True

class ForwardFirstGreedySelector(IterativeSelector):
    def __init__(self, target, continuous=[], discrete=[], ordering=[]):
        IterativeSelector.__init__(self, target, continuous=continuous, discrete=discrete, ordering=ordering, addition=True)

class BackFirstGreedySelector(IterativeSelector):
    def __init__(self, target, continuous=[], discrete=[], ordering=[]):
        IterativeSelector.__init__(self, target, continuous=continuous, discrete=discrete, ordering=ordering, addition=False)

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

import numpy as np
from collections import defaultdict

def summarise_results(results, ascending=True):
    averaged_results = {key: np.average(value) for key,value in results.items()}
    summ = defaultdict(float)
    for k,v in averaged_results.items():
        for s in k.split(","):
            summ[s] += float(v)

    return sorted(summ.items(), key=lambda x: x[1], reverse=(not ascending))

def summarise_best_combinations(results):
    averaged_results = {key: np.average(value) for key,value in results.items()}
    return sorted(averaged_results.items(), key=lambda x: x[1], reverse=True)

from collections import OrderedDict
from sklearn.cross_validation import KFold
class VariableSelectionWrapper:

    def __init__(self, selector, score_func, logger):
        self._selector = selector
        self._score_func =  score_func
        self._logger = logger
        self._models = []

    def pick_vars(self, data, n_folds=3):


        kf = KFold(data.shape[0], n_folds=n_folds, shuffle=True)
        results = OrderedDict()

        network_factory = NetworkFactory(data, self._logger)
        self._logger.debug("Written dataset")

        i = 0
        while self._selector.next_combination():

            self._logger.debug("Combination: {}".format( ",".join(self._selector.get_key_variables())))

            network = network_factory.create(discrete=list(self._selector.get_discrete_variables()), continuous=list(self._selector.get_continuous_variables()))
            key = ",".join(self._selector.get_key_variables())

            results.update({ key: [] })
            for k, (train_indexes, test_indexes) in enumerate(kf):

                _, X_test = data.ix[train_indexes], data.ix[test_indexes]
                trained_model = network_factory.create_trained_model(network, train_indexes)
                self._models.append(trained_model)

                r = trained_model.predict(test_indexes, targets=[self._selector.target])

                score = self._score_func(X_test[self._selector.target], r[self._selector.target])
                self._logger.debug("Score i={}, k={}: {}".format( i, k, score))
                results[key].append(score)
            i += 1

        return results