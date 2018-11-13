import pandas as pd
import uuid
from bayesianpy.jni import *
from bayesianpy.data import DataFrame
import bayesianpy.data
import os
from typing import List, Tuple
import numpy as np
from typing import Iterator, Optional
from . import distributed as dk

def create_network():
    return bayesServer().Network(str(uuid.getnode()))


def create_network_from_file(path, encoding='utf-8'):
    network = create_network()

    if encoding != 'utf-8':
        with open(path, mode='r', encoding=encoding) as fh:
            str = fh.read()
            network.loadFromString(str)
    else:
        network.load(path)

    return network


def create_network_from_string(network_string):
    network = create_network()
    network.loadFromString(network_string)
    return network


STATE_DELIMITER = "$$"


def state(variable, state):
    return "{0}{1}{2}".format(variable, STATE_DELIMITER, state)


class Discrete:
    def __init__(self, variable, state):
        self.variable = variable
        self.state = state

    def tostring(self):
        return state(self.variable, self.state)

    @staticmethod
    def fromstring(text):
        return Discrete(*text.split(STATE_DELIMITER))

    def __str__(self):
        return self.tostring()


def get_node(network, node):
    return network.getNodes().get(node)


def get_variable_from_node(node):
    return node.getVariables()[0]


class NetworkLinks:
    def __init__(self, network, links):
        self._links = links
        self._network = network

    def delete_between(self, a:str, b:str):
        Builder.delete_link(self._network, a, b)

    def __len__(self):
        return len(self._links)


class NetworkVariables:
    def __init__(self, network, variables):
        self._network = network
        self._variables = variables

    def get(self, name) -> 'NetworkVariable':
        return NetworkVariable(self._network, self._network.getVariables().get(name))

    def discrete(self) -> 'NetworkVariables':
        return NetworkVariables(self._network, get_discrete_variables(self._network))

    def continuous(self) -> 'NetworkVariables':
        return NetworkVariables(self._network, get_continuous_variables(self._network))

    def __getitem__(self, item) -> 'NetworkVariable':
        return self.get(item)

    def __iter__(self) -> Iterator['NetworkVariable']:
        yield from [NetworkVariable(self._network, variable) for variable in self._variables]

    def __len__(self):
        return len(self._variables)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in set(node.getName() for node in self._variables)

        if isinstance(item, NetworkVariable):
            return item.name() in [node.getName() for node in self._variables]

        if isinstance(item, jp.JClass):
            return item.getName() in [node.getName() for node in self._variables]

    def first(self) -> 'NetworkVariable':
        return NetworkVariable(self._network, self._variables.get(0))


class Buildable(object):
    def build(self):
        pass


class NetworkNode:
    def __init__(self, network, node):
        self._node = node
        self._network = network

    def name(self):
        return self._node.getName()

    def variables(self) -> 'NetworkVariables':
        return NetworkVariables(self._network, self._node.getVariables())

    def variable(self) -> 'NetworkVariable':
        v = self._node.getVariables()
        if len(v) > 1:
            raise ValueError("There were multiple variables associated with the node")

        return NetworkVariable(self._network, v.get(0))

    def links(self) -> 'NetworkLinks':
        node = self._node
        return NetworkLinks(self._network, node.getLinks())

    def parents(self) -> 'NetworkNodes':
        node = self._node
        return NetworkNodes(self._network, [link.getTo() for link in node.getLinksIn()])

    def children(self) -> 'NetworkNodes':
        node = self._node
        return NetworkNodes(self._network, [link.getTo() for link in node.getLinksOut()])


    def type(self) -> str:
        if len(self.variables()) == 1:
            return self.variables().first().type()
        if all(v.type() == NetworkVariable.Continuous for v in self.variables()):
            return NetworkVariable.Continuous
        if all(v.type() == NetworkVariable.Discrete for v in self.variables()):
            return NetworkVariable.Discrete

        return "hybrid"

    def __str__(self):
        return "{} [{}]".format(self.name(), self.type())


class VariableStates:

    def __init__(self, network, states):
        self._network = network
        self._states = states

    def get(self, name: str) -> jp.JClass:
        return self._states.get(name)

    def __getitem__(self, item):
        return self.get(item)

    def __iter__(self) -> Iterator[NetworkNode]:
        yield from [NetworkNode(self._network, node) for node in self._states]

    def __len__(self):
        return len(self._states)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in set(node.getName() for node in self._states)

        #if isinstance(item, NetworkState):
        #    return item.name() in [node.getName() for node in self._nodes]

        if isinstance(item, type(jp.JClass)):
            return item.getName() in [node.getName() for node in self._states]


class NetworkVariable:
    Discrete = "discrete"
    Continuous = "continuous"

    DoubleInterval = "double_interval"
    Boolean = "boolean"
    Integer = "integer"

    def __init__(self, network, variable):
        self._network = network
        self._variable = variable

    def __str__(self):
        return self.name()

    def name(self):
        return self._variable.getName()

    def type(self):
        if bayesServer().VariableValueType.DISCRETE == self._variable.getValueType():
            return self.Discrete

        if bayesServer().VariableValueType.CONTINUOUS == self._variable.getValueType():
            return self.Continuous

    def number_of_states(self) -> Optional[int]:
        if self.is_discrete():
            return len(self.states())

        return None

    def states(self) -> VariableStates:
        return VariableStates(self._network, self._variable.getStates())

    def is_continuous(self):
        return self.type() == self.Continuous

    def is_discrete(self):
        return self.type() == self.Discrete

    def is_discretised(self):
        return self.is_discrete() and self.state_type() == self.DoubleInterval

    def is_boolean(self):
        return self.is_discrete() and len(self.states()) == 2

    def is_single_state(self):
        return self.is_discrete() and len(self.states()) == 1

    def state_type(self):
        state_value_type = self._variable.getStateValueType()
        if bayesServer().StateValueType.DOUBLE_INTERVAL == state_value_type:
            return __class__.DoubleInterval

        if bayesServer().StateValueType.BOOLEAN == state_value_type:
            return __class__.Boolean

        if bayesServer().StateValueType.INTEGER == state_value_type:
            return __class__.Integer

        if bayesServer().StateValueType.NONE == state_value_type:
            return None


class NetworkNodes:
    def __init__(self, network, nodes):
        self._nodes = nodes
        self._network = network

    def delete(self, name: str):
        remove_node(self._network, name)

    def get(self, name: str) -> NetworkNode:
        return NetworkNode(self._network, get_node(self._network, name))

    #def add(self, node_names: List[str]):
    #    return NetworkNodeBuilder(self._network, node_names)

    def has_distributions(self) -> bool:
        return all(node.getDistribution() != None for node in self._nodes)

    def __getitem__(self, item):
        return self.get(item)

    def __iter__(self) -> Iterator[NetworkNode]:
        yield from [NetworkNode(self._network, node) for node in self._nodes]

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in set(node.getName() for node in self._nodes)

        if isinstance(item, NetworkNode):
            return item.name() in [node.getName() for node in self._nodes]

        if isinstance(item, jp.JClass):
            return item.getName() in [node.getName() for node in self._nodes]

    def __str__(self):
        return ", ".join([node.__str__() for node in self])


class Network(object):
    def __init__(self, network):
        self._network = network

    def to_xml(self):
        from xml.dom import minidom
        nt = self._network.saveToString()
        reparsed = minidom.parseString(nt)
        return reparsed.toprettyxml(indent="  ")

    def to_string(self):
        return self._network.saveToString()

    @staticmethod
    def from_new():
        return Network(create_network())

    @staticmethod
    def from_file(network_path: str, encoding='utf8'):
        return Network(create_network_from_file(network_path, encoding))

    @staticmethod
    def from_string(self, network_string: str):
        return Network(create_network_from_string(network_string))

    def links(self) -> NetworkLinks:
        return NetworkLinks(self._network, self._network.getLinks())

    def nodes(self) -> NetworkNodes:
        return NetworkNodes(self._network, self._network.getNodes())

    def variables(self) -> NetworkVariables:
        return NetworkVariables(self._network, self._network.getVariables())

    def save(self, path):
        save(self._network, path)

    def jclass(self) -> jp.JClass:
        return self._network

    def __str__(self):
        return self.to_string()


class Builder:
    @staticmethod
    def get_variable(network, variable):
        return network.getVariables().get(variable)

    @staticmethod
    def try_get_node(network, node_name):
        try:
            n = Builder.get_node(network, node_name)
            return n
        except:
            return False

    @staticmethod
    def get_node(network, node):
        return get_node(network, node)

    @staticmethod
    def delete_link(network, n1, n2):
        if isinstance(n1, str):
            n1_name = n1
            n1 = Builder.get_node(network, n1)

        if isinstance(n2, str):
            n2_name = n2
            n2 = Builder.get_node(network, n2)

        if n1 is None:
            raise ValueError("N1 {} was not recognised".format(n1_name))

        if n2 is None:
            raise ValueError("N2 {} was not recognised".format(n2_name))

        to_remove = None
        for l in n1.getLinksOut():
            if l.getTo() == n2:
                to_remove = l

        if to_remove is not None:
            network.getLinks().remove(to_remove)

    @staticmethod
    def delete_links_from(network, node):
        if isinstance(node, str):
            node = Builder.get_node(network, node)

        for link in list(node.getLinksOut()):
            network.getLinks().remove(link)

    @staticmethod
    def delete_links_to(network, node):
        if isinstance(node, str):
            node = Builder.get_node(network, node)

        for link in list(node.getLinksIn()):
            network.getLinks().remove(link)

    @staticmethod
    def create_link(network, n1, n2, t=None):
        if isinstance(n1, str):
            n1_name = n1
            n1 = Builder.get_node(network, n1)

        if isinstance(n2, str):
            n2_name = n2
            n2 = Builder.get_node(network, n2)

        if n1 is None:
            raise ValueError("N1 {} was not recognised".format(n1_name))

        if n2 is None:
            raise ValueError("N2 {} was not recognised".format(n2_name))

        if t is not None:
            l = bayesServer().Link(n1, n2, t)
        else:
            l = bayesServer().Link(n1, n2)

        try:
            network.getLinks().add(l)
        except BaseException as e:
            raise ValueError(e.message() + ". Trying to add link from {} to {}".format(n1.getName(), n2.getName()))

    @staticmethod
    def _create_interval_name(interval, decimal_places):
        title = ""
        title += "(" if interval.getMinimumEndPoint() == bayesServer().IntervalEndPoint.OPEN else "["
        title += "{0:.{digits}f},{1:.{digits}f}".format(interval.getMinimum().floatValue(),
                                                        interval.getMaximum().floatValue(), digits=decimal_places)
        title += ")" if interval.getMaximumEndPoint() == bayesServer().IntervalEndPoint.OPEN else "]"
        return title


    @staticmethod
    def create_utility_node(network, node_name):
        n = Builder.try_get_node(network, node_name)
        if n is not None:
            return n

        v = bayesServer().Variable(node_name, bayesServer().VariableValueType.CONTINUOUS, bayesServer().VariableKind.Utility)
        n_ = bayesServer().Node(v)

        network.getNodes().add(n_)

        return n_

    @staticmethod
    def create_discretised_variables(network, data, node_names, bin_count=4, infinite_extremes=True,
                                     decimal_places=4, mode='EqualFrequencies',
                                     zero_crossing=True, defined_bins: List[Tuple[float, float]] = None,):
        node_names = [str(name) for name in node_names]
        if defined_bins is None:
            options = bayesServerDiscovery().DiscretizationOptions()
            options.setInfiniteExtremes(infinite_extremes)
            options.setSuggestedBinCount(bin_count)

            # reads data from either a Pandas dataframe or dask, so will support out of memory and in-memory.
            data_reader_cmd = bayesianpy.data.DaskDataset(data[node_names]).create_data_reader_command().create()

            if mode == 'EqualFrequencies':
                ef = bayesServerDiscovery().EqualFrequencies()
            elif mode == 'EqualIntervals':
                ef = bayesServerDiscovery().EqualIntervals()
            else:
                raise ValueError("mode not recognised")

            columns = jp.java.util.Arrays.asList(
                [bayesServerDiscovery().DiscretizationColumn(name) for name in node_names])
            column_intervals = ef.discretize(data_reader_cmd, columns,
                                             bayesServerDiscovery().DiscretizationAlgoOptions())

            for i, interval in enumerate(column_intervals):

                intervals = list(interval.getIntervals().toArray())
                if zero_crossing:
                    end_point_value = 0.5

                    zero = bayesServer().Interval(jp.java.lang.Double(jp.java.lang.Double.NEGATIVE_INFINITY),
                                                  jp.java.lang.Double(end_point_value),
                                                  bayesServer().IntervalEndPoint.CLOSED,
                                                  bayesServer().IntervalEndPoint.OPEN)

                    if 0.5 < intervals[0].getMaximum().floatValue():
                        # if the interval starts and ends at end_point_value then remove it
                        if intervals[0].getMaximum() == end_point_value:
                            intervals.pop(0)
                        else:
                            intervals[0].setMinimum(jp.java.lang.Double(0.5))
                            intervals[0].setMinimumEndPoint(bayesServer().IntervalEndPoint.CLOSED)

                        intervals = [zero] + intervals

                v = bayesServer().Variable(node_names[i], bayesServer().VariableValueType.DISCRETE)
                v.setStateValueType(bayesServer().StateValueType.DOUBLE_INTERVAL)
                n = bayesServer().Node(v)
                for interval in intervals:
                    v.getStates().add(
                          bayesServer().State("{}".format(Builder._create_interval_name(interval, decimal_places)),
                                        interval))

                network.getNodes().add(n)
                yield n

        else:
            for node in node_names:
                intervals = []
                for bin in defined_bins:
                    minEndPoint = bayesServer().IntervalEndPoint.CLOSED
                    maxEndPoint = bayesServer().IntervalEndPoint.OPEN

                    if np.isneginf(float(bin[0])):
                        a = jp.java.lang.Double(jp.java.lang.Double.NEGATIVE_INFINITY)
                    else:
                        a = jp.java.lang.Double(bin[0])

                    if np.isposinf(float(bin[1])):
                        b = jp.java.lang.Double(jp.java.lang.Double.POSITIVE_INFINITY)
                    else:
                        b = jp.java.lang.Double(bin[1])

                    intervals.append(
                        bayesServer().Interval(a, b, minEndPoint,
                                               maxEndPoint))

                v = bayesServer().Variable(node, bayesServer().VariableValueType.DISCRETE)
                v.setStateValueType(bayesServer().StateValueType.DOUBLE_INTERVAL)
                n = bayesServer().Node(v)
                for interval in intervals:
                    v.getStates().add(
                        bayesServer().State("{}".format(Builder._create_interval_name(interval, decimal_places)),
                                            interval))

                network.getNodes().add(n)
                yield n


    @staticmethod
    def create_discretised_variable(network, data, node_name, bin_count=4,
                                    infinite_extremes=True,
                                    decimal_places=4,
                                    mode='EqualFrequencies',
                                    bins=[], zero_crossing=False):
        node_name = str(node_name)
        if len(bins) == 0:
            options = bayesServerDiscovery().DiscretizationOptions()
            options.setInfiniteExtremes(infinite_extremes)
            options.setSuggestedBinCount(bin_count)

            # reads data from either a Pandas dataframe or dask, so will support out of memory and in-memory.
            data_reader_cmd = bayesianpy.data.DaskDataset(data[[node_name]]).create_data_reader_command().create()

            if mode == 'EqualFrequencies':
                ef = bayesServerDiscovery().EqualFrequencies()
            elif mode == 'EqualIntervals':
                ef = bayesServerDiscovery().EqualIntervals()
            else:
                raise ValueError("mode not recognised")

            # TODO: currently just looking at a single column at a time, which isn't very efficient.
            columns = jp.java.util.Arrays.asList([bayesServerDiscovery().DiscretizationColumn(node_name)])
            intervals = ef.discretize(data_reader_cmd, columns,
                                      bayesServerDiscovery().DiscretizationAlgoOptions()) \
                .get(0).getIntervals()

            if zero_crossing:
                end_point_value = 0.5
                intervals = list(intervals.toArray())
                zero = bayesServer().Interval(jp.java.lang.Double(jp.java.lang.Double.NEGATIVE_INFINITY),
                                              jp.java.lang.Double(end_point_value), bayesServer().IntervalEndPoint.CLOSED,
                                              bayesServer().IntervalEndPoint.OPEN)

                if 0.5 < intervals[0].getMaximum().floatValue():
                    # if the interval starts and ends at end_point_value then remove it
                    if intervals[0].getMaximum() == end_point_value:
                        intervals.pop(0)
                    else:
                        intervals[0].setMinimum(jp.java.lang.Double(0.5))
                        intervals[0].setMinimumEndPoint(bayesServer().IntervalEndPoint.CLOSED)

                    intervals = [zero] + intervals
        else:
            intervals = []
            for bin in bins:
                minEndPoint = bayesServer().IntervalEndPoint.CLOSED if bin[
                                                                           2] == "closed" else bayesServer().IntervalEndPoint.OPEN
                maxEndPoint = bayesServer().IntervalEndPoint.CLOSED if bin[
                                                                           3] == "closed" else bayesServer().IntervalEndPoint.OPEN
                intervals.append(
                    bayesServer().Interval(jp.java.lang.Double(bin[0]), jp.java.lang.Double(bin[1]), minEndPoint,
                                           maxEndPoint))

        v = bayesServer().Variable(node_name, bayesServer().VariableValueType.DISCRETE)
        v.setStateValueType(bayesServer().StateValueType.DOUBLE_INTERVAL)
        n = bayesServer().Node(v)
        for interval in intervals:
            v.getStates().add(
                bayesServer().State("{}".format(Builder._create_interval_name(interval, decimal_places)), interval))

        network.getNodes().add(n)
        return n


    @staticmethod
    def create_continuous_variable(network, node_name):
        n = Builder.try_get_node(network, node_name)
        if n is not None:
            return n

        v = bayesServer().Variable(node_name, bayesServer().VariableValueType.CONTINUOUS)
        n_ = bayesServer().Node(v)

        network.getNodes().add(n_)

        return n_


    @staticmethod
    def create_cluster_variable(network, num_states, variable_name='Cluster'):
        v = bayesServer().Variable(variable_name)
        parent = bayesServer().Node(v)
        for i in range(num_states):
            v.getStates().add(bayesServer().State("Cluster{}".format(i)))

        network.getNodes().add(parent)
        return parent


    @staticmethod
    def create_multivariate_continuous_node(network, variables, node_name):
        n_ = bayesServer().Node(node_name,
                                [bayesServer().Variable(v, bayesServer().VariableValueType.CONTINUOUS) for v in variables])
        network.getNodes().add(n_)
        return n_


    @staticmethod
    def create_discrete_variable(network, df: pd.DataFrame, node_name: str, states: List[str] = None, blanks=None):
        n = Builder.try_get_node(network, node_name)
        if n is not None:
            return n

        v = bayesServer().Variable(node_name)
        n_ = bayesServer().Node(v)

        if states is None:
            states = dk.compute(df[str(node_name)].dropna().unique()).tolist()

        for s in states:
            v.getStates().add(bayesServer().State(str(s)))

        if node_name in df.columns.tolist():

            if DataFrame.is_int(df[str(node_name)].dtype) or DataFrame.could_be_int(df[str(node_name)]):
                v.setStateValueType(bayesServer().StateValueType.INTEGER)
                for state in v.getStates():
                    state.setValue(jp.java.lang.Integer(int(float(state.getName()))))

            if DataFrame.is_bool(df[str(node_name)].dtype):
                v.setStateValueType(bayesServer().StateValueType.BOOLEAN)
                for state in v.getStates():
                    state.setValue(state.getName() == 'True')

        network.getNodes().add(n_)

        return n_


def get_node_names(nt):
    return [node.getName() for node in nt.getNodes()]


def is_variable_discrete(v):
    return v.getValueType() == bayesServer().VariableValueType.DISCRETE


def is_variable_continuous(v):
    return v.getValueType() == bayesServer().VariableValueType.CONTINUOUS


def is_variable_discretised(v):
    return v.getStateValueType() == bayesServer().StateValueType.DOUBLE_INTERVAL

def interval_is_between(value, interval):
    min_value = interval.getMinimum()
    max_value = interval.getMaximum()

    if min_value == jp.java.lang.Double.NEGATIVE_INFINITY:
        min_value = -np.inf
    else:
        min_value = min_value.floatValue()

    if max_value == jp.java.lang.Double.POSITIVE_INFINITY:
        max_value = np.inf
    else:
        max_value = max_value.floatValue()

    max_endpoint = interval.getMaximumEndPoint()
    min_endpoint = interval.getMinimumEndPoint()

    bs_closed = bayesServer().IntervalEndPoint.CLOSED
    bs_open = bayesServer().IntervalEndPoint.OPEN

    if min_endpoint == bs_closed and max_endpoint == bs_open:
        return min_value < value <= max_value
    if min_endpoint == bs_closed and max_endpoint == bs_closed:
        return min_value < value < max_value
    if min_endpoint == bs_open and max_endpoint == bs_closed:
        return min_value <= value < max_value
    else:
        return min_value <= value <= max_value


def get_variable(network, variable_name):
    variable = network.getVariables().get(variable_name)
    if variable is None:
        raise ValueError("Variable {} does not exist".format(variable_name))

    return variable


def variable_exists(network, variable_name):
    try:
        get_variable(network, variable_name)
        return True
    except ValueError:
        return False


def remove_continuous_nodes(network):
    n = network.copy()
    to_remove = []
    for v in n.getVariables():
        if is_variable_continuous(v):
            to_remove.append(v)

    for v in to_remove:
        node = v.getNode()
        n.getNodes().remove(node)

    return n


def remove_single_state_nodes(network):
    to_remove = []
    for node in get_nodes(network):

        v = get_variable_from_node(node)
        if is_variable_discrete(v):
            if len(v.getStates()) <= 1:
                to_remove.append(node)

    for node in to_remove:
        remove_node(network, node)

    return network


def get_nodes(network):
    for node in network.getNodes():
        yield node


def get_continuous_nodes(network):
    for node in network.getNodes():
        if bayesianpy.network.is_variable_continuous(node.getVariables().get(0)):
            yield node


def get_continuous_variables(network):
    for variable in network.getVariables():
        if bayesianpy.network.is_variable_continuous(variable):
            yield variable


def get_discrete_variables(network):
    for variable in network.getVariables():
        if bayesianpy.network.is_variable_discrete(variable):
            yield variable


def remove_node(network, node):
    if node is None:
        raise ValueError("Node must be specified when trying to remove it.")
    network.getNodes().remove(node)


def get_number_of_states(network, variable):
    v = network.getVariables().get(variable)
    return len(v.getStates())


def get_state(network, variable_name, state_name):
    variable = get_variable(network, variable_name)
    for jstate in variable.getStates():
        if jstate.getName() == str(state_name):
            return jstate


def get_other_states_from_variable(network, target):
    target_ = network.getVariables().get(target.variable)
    for st in target_.getStates():
        if st.getName() == str(target.state):
            continue

        yield state(target.variable, st.getName())


def create_variable_references(network, data, variable_references=[]):
    """
    Match up network variables to the dataframe columns
    :param data: dataframe
    :return: a list of 'VariableReference' objects
    """

    variables = []

    if len(variable_references) == 0:
        variables = network.getVariables()
    else:
        for v in variable_references:
            variables.append(get_variable(network, v))

    latent_variable_name = "Cluster"
    for v in variables:
        #if v.getName().startswith(latent_variable_name):
        #    continue

        if v.getName() not in data.columns.tolist():
            continue

        name = v.getName()

        valueType = bayesServer().data.ColumnValueType.VALUE

        if v.getStateValueType() == bayesServer().StateValueType.NONE:
            valueType = bayesServer().data.ColumnValueType.NAME
        elif v.getStateValueType() != bayesServer().StateValueType.DOUBLE_INTERVAL \
                and is_variable_discrete(v):

            if not DataFrame.is_int(data[name].dtype) and not DataFrame.is_bool(data[name].dtype) \
                    and not DataFrame.is_float(data[name].dtype):
                valueType = bayesServer().data.ColumnValueType.NAME

        yield bayesServer().data.VariableReference(v, valueType, name,
                                                   bayesServer().data.StateNotFoundAction.MISSING_VALUE)


def save(network, path):
    from xml.dom import minidom
    nt = network.saveToString()
    reparsed = minidom.parseString(nt)
    with open(path, 'w') as fh:
        fh.write(reparsed.toprettyxml(indent="  "))


def is_cluster_variable(v):
    if not isinstance(v, str):
        v = v.getName()
    return v == "Cluster" or v.startswith("Cluster_")


def is_trained(network):
    for n in network.getNodes():
        if n.getDistribution() is None:
            return False

    return True


class NetworkFactory:
    def __init__(self, logger, network_file_path=None, network=None, encoding='utf-8'):
        self._logger = logger
        self._network_file_path = network_file_path
        self._network = network
        self._encoding = encoding

    def create_from_file(self, path):
        return create_network_from_file(path, self._encoding)

    def create(self):
        if self._network is not None:
            return self._network
        elif self._network_file_path is None or not os.path.exists(self._network_file_path):
            return create_network()
        else:
            return self.create_from_file(self._network_file_path)
