from typing import Dict, List
import bayesianpy.network
from bayesianpy.jni import jp
import numpy as np

class DiscreteSpecification:
    def __init__(self, state_values, is_flexible=True, distance_values: Dict[str, int]=None):
        if isinstance(state_values, str):
            state_values = [state_values]

        self.distance_values = distance_values
        self._state_values = state_values
        self.is_flexible = is_flexible

    def items(self):
        return self._state_values

    def __contains__(self, item):
        return item in self._state_values


class DiscreteJointDistributionSpecification:
    def __init__(self, specification_container: Dict[str, 'DiscreteSpecification']):
        self.spec = specification_container

    def _distance(self, a, i: 'DiscreteSpecification') -> int:
        values = i.distance_values
        a_v = values[a]

        distances = []
        for item in i.items():
            i_v = values[item]
            distances.append(abs(a_v - i_v))

        return min(distances)

    def max_distance(self) -> int:
        return len(self.spec) * 2

    def distance(self, variables: Dict[str, str]) -> int:
        d = 0
        for k, v in variables.items():

            if not self.spec[k].is_flexible and v not in self.spec[k]:
                return self.max_distance()

            d += self._distance(v, self.spec[k])

        return d

class TableAccessor:

    def __init__(self, node, distribution):
        self._node = node
        self._dist = distribution
        self._accessor = get_table_accessor(node, distribution)

    def get_probability_for(self, variable_states: Dict[str,str]) -> float:
        return get_probabilities_from_accessor(self._node, variable_states, self._accessor)

    def get_probabilities_for(self, list_of_variable_states: List[Dict[str, str]]) -> List[float]:
        return [self.get_probability_for(variable_states)
                for variable_states in list_of_variable_states]

    def get_total_probability_for(self, list_of_variable_states: List[Dict[str, str]]) -> float:
        return np.sum(self.get_probabilities_for(list_of_variable_states))


def normalize(node) -> None:
    """
    Normalizes the distribution depending on how many parents there are
    :param node: Java Node object
    """
    ti = TableIterator(node)
    while ti.read():
        ti.set_normalized_value()

def set_probability_on_divorcing_node(iterator: 'TableIterator') -> None:

    """
    Set the probability of distribution for a divorcing node based on the parent nodes. Assumes
    that the node has same states as parent nodes.
    :param iterator: TableIterator
    :return: None
    """

    while iterator.read():
        matches = len([item for item in iterator.get_parent_state_names()
                       if item == iterator.get_node_state_name()])
        value = (1.0 / (len(iterator.get_parent_state_names()))) * matches
        iterator.set_value(value)

def set_remainder_probability(iterator: 'TableIterator', accessor: 'TableAccessor') -> None:

    """
    Sets the remainder for a combination (e.g. p-1)
    :param iterator: TableIterator
    :param accessor: TableAccessor
    :return: None
    """

    p = accessor.get_total_probability_for(iterator.get_possible_combinations())
    iterator.set_value(1 - p)

def create_table_accessor(table_iterator: 'TableIterator'):
    return TableAccessor(table_iterator.get_node(), table_iterator.get_distribution())


class TableIterator:

    def __init__(self, node):
        self._node = node
        self._dist = create_distribution(node)
        self._iterator = get_table_iterator(node, self._dist)
        self._i = 0

    def get_node(self):
        return self._node

    def get_distribution(self) -> object:
        return self._dist

    def get_state_names(self) -> List[str]:
        return get_state_names_from_iterator(self._node, self._iterator)

    def get_parent_state_names(self) -> List[str]:
        state_names = get_state_names_from_iterator(self._node, self._iterator)
        return state_names[0 - (len(state_names) - 1):]

    def get_state_indexes(self) -> List[int]:
        return get_state_indexes_from_iterator(self._node, self._iterator)

    def set_value(self, value:float) -> None:
        self._iterator.setValue(value)

    def set_value_or_remainder(self, value:float, accessor: 'TableAccessor') -> None:
        if self.is_remainder():
            value = 1 - np.sum(accessor.get_probabilities_for(self.get_possible_combinations()))

        self.set_value(value)

    def set_normalized_value(self) -> None:
        total = len(bayesianpy.network.get_variable_from_node(self._node).getStates())
        self.set_value(1 / total)

    def get_node_order(self) -> List[object]:
        return get_node_order(self._node)

    def get_variable_state_names(self) -> Dict[str, str]:
        state_names = self.get_state_names()
        return {variable.getName(): state_names[i] for i, variable in enumerate(get_node_order(self._node))}

    def get_parent_variable_state_names(self) -> Dict[str, str]:
        variables = self.get_variable_state_names()
        variables.pop(self._node.getName())
        return variables

    def is_remainder(self) -> bool:
        """
        Checks whether the current table entry is the last one for the target combination (e.g. requires the
        remainder from 1).
        :return: bool
        """
        return self.get_node_state_index() == len(bayesianpy.network.get_variable_from_node(self._node)
                                                        .getStates()) - 1

    def get_possible_combinations(self) -> List[Dict[str, str]]:
        current_combination = self.get_parent_variable_state_names()
        combinations = []
        for state in bayesianpy.network.get_variable_from_node(self._node).getStates():
            combinations.append({**{self._node.getName(): state.getName()}, **current_combination})

        return combinations

    def get_node_state_name(self) -> str:
        variables = self.get_variable_state_names()
        return variables[self._node.getName()]

    def get_node_state_index(self) -> int:
        return self.get_state_indexes()[0]

    def set_table(self) -> None:
        self._node.setDistribution(self._iterator.getTable())

    def __next__(self) -> 'TableIterator':
        if self.read():
            return self
        else:
            raise StopIteration

    def read(self) -> bool:
        if self._i > 0:
            self._iterator.increment()

        if self._i < self._iterator.size():
            self._i += 1
            return True

        self.set_table()
        return False

    def __iter__(self) -> 'TableIterator':
        return self


def get_node_order(node) -> List[object]:
    return [node] + [link.getFrom() for link in node.getLinksIn()]

def create_distribution(node) -> object:
    return node.newDistribution()

def get_table_iterator(node, distribution=None) -> object:

    if distribution is None:
        distribution = create_distribution(node)

    variable_order = get_node_order(node)
    ti = bayesianpy.jni.bayesServer().TableIterator(distribution, variable_order)

    return ti

def get_table_accessor(node, distribution=None) -> object:
    if distribution is None:
        distribution = create_distribution(node)

    variable_order = get_node_order(node)
    ta = bayesianpy.jni.bayesServer().TableAccessor(distribution, variable_order)

    return ta

def get_state_indexes_from_iterator(node, table_iterator) -> List[int]:
    JavaIntArray = jp.JArray(jp.JInt)
    state_order = JavaIntArray([0] * len(get_node_order(node)))
    table_iterator.getStates(state_order)

    return list(state_order)

def get_state_names_from_iterator(node, table_iterator) -> List[str]:
    variable_order = get_node_order(node)
    state_order = get_state_indexes_from_iterator(node, table_iterator)
    state_names = [bayesianpy.network.get_variable_from_node(variable_order[variable_index]).getStates().
                       get(state_index).getName()
                   for variable_index, state_index in enumerate(state_order)]

    return list(state_names)

def get_probabilities_from_accessor(node, states:Dict[str, str], table_accessor) -> List[float]:
    variable_order = get_node_order(node)
    JavaIntArray = jp.JArray(jp.JInt)
    indexes = JavaIntArray([[state.getName() for state in bayesianpy.network.get_variable_from_node(n).getStates()].index(states[n.getName()])
                 for n in variable_order])

    return table_accessor.get(indexes)
