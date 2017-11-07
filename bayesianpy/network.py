import pandas as pd
import uuid
from bayesianpy.jni import *
from bayesianpy.data import DataFrame
import os
from typing import List
import numpy as np

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

def create_network_from_string(path):
    network = create_network()
    network.loadFromString(path)
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
        title += "{0:.{digits}f},{1:.{digits}f}".format(interval.getMinimum().floatValue(), interval.getMaximum().floatValue(), digits=decimal_places)
        title += ")" if interval.getMaximumEndPoint() == bayesServer().IntervalEndPoint.OPEN else "]"
        return title

    @staticmethod
    def create_discretised_variable(network, data, node_name, bin_count=4,
                                    infinite_extremes=True,
                                    decimal_places=4,
                                    mode='EqualFrequencies',
                                    bins=[], zero_crossing=False):
        if len(bins) == 0:
            options = bayesServerDiscovery().DiscretizationOptions()
            options.setInfiniteExtremes(infinite_extremes)
            options.setSuggestedBinCount(bin_count)
            values = jp.java.util.Arrays.asList(data[node_name].astype(float).dropna().tolist())
            if mode == 'EqualFrequencies':
                ef = bayesServerDiscovery().EqualFrequencies()
            elif mode == 'EqualIntervals':
                ef = bayesServerDiscovery().EqualIntervals()
            else:
                raise ValueError("mode not recognised")

            intervals = ef.discretize(values, options, jp.JString(node_name))
            if zero_crossing:
                end_point_value = 0.5
                intervals = list(intervals.toArray())
                zero = bayesServer().Interval(jp.java.lang.Double(jp.java.lang.Double.NEGATIVE_INFINITY), jp.java.lang.Double(end_point_value), bayesServer().IntervalEndPoint.CLOSED,
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
                minEndPoint = bayesServer().IntervalEndPoint.CLOSED if bin[2] == "closed" else bayesServer().IntervalEndPoint.OPEN
                maxEndPoint = bayesServer().IntervalEndPoint.CLOSED if bin[3] == "closed" else bayesServer().IntervalEndPoint.OPEN
                intervals.append(bayesServer().Interval(jp.java.lang.Double(bin[0]), jp.java.lang.Double(bin[1]), minEndPoint, maxEndPoint))

        v = bayesServer().Variable(node_name, bayesServer().VariableValueType.DISCRETE)
        v.setStateValueType(bayesServer().StateValueType.DOUBLE_INTERVAL)
        n = bayesServer().Node(v)
        for interval in intervals:
            v.getStates().add(bayesServer().State("{}".format(Builder._create_interval_name(interval, decimal_places)), interval))

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
        n_ = bayesServer().Node(node_name, [bayesServer().Variable(v, bayesServer().VariableValueType.CONTINUOUS) for v in variables])
        network.getNodes().add(n_)
        return n_

    @staticmethod
    def create_discrete_variable(network, df: pd.DataFrame, node_name: str, states:List[str]=None, blanks=None):
        n = Builder.try_get_node(network, node_name)
        if n is not None:
            return n
            
        v = bayesServer().Variable(node_name)
        n_ = bayesServer().Node(v)

        if states is None:
            if blanks is not None:
                states = df[node_name].replace(np.nan, blanks).unique()
            else:
                states = df[node_name].replace("", np.nan).dropna().unique()

        for s in states:
            v.getStates().add(bayesServer().State(str(s)))

        if node_name in df.columns.tolist():

            if DataFrame.is_int(df[node_name].dtype) or DataFrame.could_be_int(df[node_name]):
                v.setStateValueType(bayesServer().StateValueType.INTEGER)
                for state in v.getStates():
                    state.setValue(jp.java.lang.Integer(int(float(state.getName()))))

            if DataFrame.is_bool(df[node_name].dtype):
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
            variables.append(bayesianpy.network.get_variable(network, v))

    latent_variable_name = "Cluster"
    for v in variables:
        if v.getName().startswith(latent_variable_name):
            continue

        if v.getName() not in data.columns:
            continue

        name = v.getName()

        valueType = bayesServer().data.ColumnValueType.VALUE

        if v.getStateValueType() == bayesServer().StateValueType.NONE:
            valueType = bayesServer().data.ColumnValueType.NAME
        elif v.getStateValueType() != bayesServer().StateValueType.DOUBLE_INTERVAL \
                and bayesianpy.network.is_variable_discrete(v):

            if not DataFrame.is_int(data[name].dtype) and not DataFrame.is_bool(data[name].dtype)\
                    and not DataFrame.is_float(data[name].dtype):
                valueType = bayesServer().data.ColumnValueType.NAME

        yield bayesServer().data.VariableReference(v, valueType, name)

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
    def __init__(self, logger, network_file_path = None, network = None, encoding='utf-8'):
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
