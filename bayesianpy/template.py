from bayesianpy.network import Builder as builder
import pandas as pd
import bayesianpy.network
from bayesianpy.jni import *
import bayesianpy.dask as dk
import logging

class Template:
    def __init__(self, discrete=pd.DataFrame(), continuous=pd.DataFrame(), label:str=None):
        self._discrete = discrete
        self._continuous = continuous
        self._label = label

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        return network_factory.create()

    def get_label(self):
        return self._label


class Tpl(Template):
    def __init__(self):
        super().__init__()

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        super().create(network_factory)


class NaiveBayes(Template):
    def __init__(self, parent_node: str, logger, discrete=pd.DataFrame(), continuous=pd.DataFrame(), discrete_states={}):
        super().__init__(discrete=discrete, continuous=continuous)
        self._discrete_states = discrete_states
        self._logger = logger
        self._parent_node = parent_node

    def create(self, network_factory):
        network = network_factory.create()

        if not dk.empty(self._continuous):
            for c_name in self._continuous.columns:
                c = builder.create_continuous_variable(network, c_name)

        if dk.empty(self._discrete):
            for d_name in self._discrete.columns:
                if d_name in self._discrete_states:
                    states = self._discrete_states[d_name]
                else:
                    states = dk.compute(self._discrete[d_name].dropna().unique()).tolist()

                try:
                    c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                except BaseException as e:
                    self._logger.warn(e)

        parent_node = builder.try_get_node(network, self._parent_node)
        if parent_node is None:
            raise ValueError("Parent node: {} not recognised".format(self._parent_node))

        for node in network.getNodes():
            if node == parent_node:
                continue
            builder.create_link(network, parent_node, node)

        return network

class MixtureNaiveBayes(Template):

    def __init__(self, logger, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10, discrete_states={}, latent_variable_name='Cluster'):
        super().__init__(discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._discrete_states = discrete_states
        self._logger = logger
        self._latent_variable_name = latent_variable_name

    def create(self, network_factory):
        network = network_factory.create()
        cluster = builder.try_get_node(network, "Cluster")
        if cluster is None:
            cluster = builder.create_cluster_variable(network, self._latent_states, variable_name=self._latent_variable_name)

        if not dk.empty(self._continuous):
            for c_name in self._continuous.columns:
                c = builder.create_continuous_variable(network, c_name)
                try:
                    builder.create_link(network, cluster, c)
                except ValueError as e:
                    self._logger.warn(e)

        if not dk.empty(self._discrete):
            for d_name in self._discrete.columns:
                if d_name in self._discrete_states:
                    states = self._discrete_states[str(d_name)]
                else:
                    states = dk.compute(self._discrete[str(d_name)].dropna().unique()).tolist()

                try:
                    c = builder.create_discrete_variable(network, self._discrete, str(d_name), states)
                
                    builder.create_link(network, cluster, c)
                except BaseException as e:
                    self._logger.warn(e)


        return network

class WithoutEdges(Template):
    def __init__(self, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10,
                 discrete_states={}, blanks=None):
        super().__init__(discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._discrete_states = discrete_states
        self._blanks = blanks

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = network_factory.create()
        #builder.create_cluster_variable(network, 5)

        if not dk.empty(self._continuous):
            for c_name in self._continuous.columns:
                builder.create_continuous_variable(network, c_name)

        if not dk.empty(self._discrete):
            for d_name in self._discrete.columns:
                builder.create_discrete_variable(network, self._discrete, d_name, blanks=self._blanks)

        network = bayesianpy.network.remove_single_state_nodes(network)

        return network

class DiscretisedMixtureNaiveBayes(Template):

    def __init__(self, logger, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10, bin_count=4,
                 binning_mode='EqualFrequencies', zero_crossing=False):
        super().__init__(discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._logger = logger
        self._bin_count = bin_count
        self._binning_mode = binning_mode
        self._zero_crossing = zero_crossing

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = network_factory.create()
        cluster = builder.create_cluster_variable(network, self._latent_states)

        if not dk.empty(self._continuous):
            for c_name in self._continuous.columns:
                c = builder.create_discretised_variable(network, self._continuous, c_name, bin_count=self._bin_count,
                                                        mode=self._binning_mode, zero_crossing=self._zero_crossing)

                builder.create_link(network, cluster, c)

        if not dk.empty(self._discrete):
            for d_name in self._discrete.columns:
                states = dk.compute(self._discrete[d_name].dropna().unique())
                c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                builder.create_link(network, cluster, c)

        return network

class AutoStructure(Template):
    def __init__(self, template, dataset:bayesianpy.data.DataSet, logger:logging.Logger, use_same_model=True,
                 engine='PC', root_node:str=None):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._logger = logger
        self._use_same_model = use_same_model
        self._links = []
        self._engine = engine
        self._root_node = root_node
        self._dataset = dataset

    def learn(self, network):

        if len(self._links) > 0 and self._use_same_model:
            return self._links

        data_reader_command = self._dataset.create_data_reader_command()

        reader_options = self._dataset.get_reader_options()
        network.getLinks().clear()

        variable_references = list(bayesianpy.network.create_variable_references(network, self._dataset.get_dataframe()))

        evidence_reader_command = bayesServer().data.DefaultEvidenceReaderCommand(data_reader_command,
                                                                                  jp.java.util.Arrays.asList(variable_references),
                                                                                  reader_options)

        if self._engine == 'PC':
            options = bayesServerStructure().PCStructuralLearningOptions()
            options.setMaximumConditional(2)
            self._logger.info("Learning structure from {} variables.".format(len(variable_references)))
            output = bayesServerStructure().PCStructuralLearning().learn(evidence_reader_command, jp.java.util.Arrays.asList(network.getNodes().toArray()),
                                                                 options)
        elif self._engine == 'TAN':
            options = bayesServerStructure().TANStructuralLearningOptions()
            options.setTarget(bayesianpy.network.get_node(network, self._root_node))
            self._logger.info("Learning structure from {} variables.".format(len(variable_references)))
            output = bayesServerStructure().TANStructuralLearning().learn(evidence_reader_command,
                                                                         jp.java.util.Arrays.asList(
                                                                             network.getNodes().toArray()),
                                                                                options)
        elif self._engine == 'Hierarchical':

            from sklearn.model_selection import KFold
            class DefaultEvidenceReaderCommandFactory:
                def __init__(self, ds:bayesianpy.data.DataSet, options):
                    #self._data_reader_command = cmd
                    self._ds = ds
                    #self._variable_references = refs
                    self._reader_options = options
                    self._kfold = None
                    self._partitions = {}

                def create(self, network):
                    variable_references = list(
                        bayesianpy.network.create_variable_references(network, self._ds.get_dataframe()))

                    return bayesServer().data.DefaultEvidenceReaderCommand(self._ds.create_data_reader_command(),
                                                                           jp.java.util.Arrays.asList(variable_references),
                                                                           self._reader_options)

                def createPartitioned(self, network,
                                      dataPartitioning,
                                      partitionCount):

                    if self._kfold is None:
                        self._kfold = KFold(n_splits=partitionCount, shuffle=False)

                    variable_references = list(
                        bayesianpy.network.create_variable_references(network, self._ds.get_dataframe()))

                    if dataPartitioning.getPartitionNumber() in self._partitions:
                        train, test = self._partitions[dataPartitioning.getPartitionNumber()]
                    else:
                        train, test = next(self._kfold.split(self._ds.data))
                        #train = train.index.tolist()
                        #test = test.index.tolist()
                        self._partitions.update({ dataPartitioning.getPartitionNumber() :
                                                      (train, test)})

                    if dataPartitioning.getMethod() == bayesServer().data.DataPartitionMethod.EXCLUDE_PARTITION_DATA:
                        print("Excluding")
                        subset = self._ds.subset(train)
                    else:
                        print("Including")
                        subset = self._ds.subset(test)

                    cmd = subset.create_data_reader_command()
                    return bayesServer().data.DefaultEvidenceReaderCommand(cmd,
                                                                           jp.java.util.Arrays.asList(
                                                                               variable_references),
                                                                           self._reader_options)

            ercf = DefaultEvidenceReaderCommandFactory(self._dataset, reader_options)
            proxy = jp.JProxy("com.bayesserver.data.EvidenceReaderCommandFactory", inst=ercf)

            options = bayesServerStructure().HierarchicalStructuralLearningOptions()
            self._logger.info("Learning structure from {} variables.".format(len(variable_references)))
            output = bayesServerStructure().HierarchicalStructuralLearning().learn(proxy,
                                                                         jp.java.util.Arrays.asList(
                                                                             network.getNodes().toArray()),
                                                                                options)

        self._logger.info("Created {} links.".format(len(output.getLinkOutputs())))

        for link in output.getLinkOutputs():
            self._links.append((link.getLink().getFrom().getName(), link.getLink().getTo().getName()))

        return self._links


    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)

        for link_from, link_to in self.learn(network):
            try:
                builder.create_link(network, link_from, link_to)
            except ValueError:
                self._logger.warning("Could not add link from {} to {}".format(link_from, link_to))

        return network

from typing import List

class WithTreeStructure(Template):
    def __init__(self, template: Template, root_node:str):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._root_node = root_node

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)

        root = bayesianpy.network.get_node(network, self._root_node)

        for node in bayesianpy.network.get_nodes(network):
            if node == root:
                continue
            builder.create_link(network, root, node)

        return network


class WithDiscretisedVariables(Template):
    def __init__(self, template: Template, logger, discretised_variables:List[str]=None, bins:List[int]=None,
                 mode='EqualFrequencies', default_bin_count:int=4, zero_crossing=False):

        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._discretised_variables = discretised_variables
        self._logger = logger
        self._bins = bins
        self._mode = mode
        self._default_bin_count = default_bin_count
        self._zero_crossing = zero_crossing

        if discretised_variables is None:
            self._discretised_variables = template._continuous

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)
        for i, var in enumerate(self._discretised_variables.columns.tolist()):
            node = builder.get_node(network, str(var))
            if node is not None:

                links_from = [link.getFrom() for link in node.getLinks() if link.getFrom().getName() != var]
                links_to = [link.getTo() for link in node.getLinks() if link.getTo().getName() != var]

                network.getNodes().remove(node)

            bin_count = self._default_bin_count if self._bins is None else self._bins[i]

            n = builder.create_discretised_variable(network, self._continuous, var,
                                                    bin_count=bin_count, mode=self._mode, zero_crossing=self._zero_crossing)

            if node is not None:
                for l in links_from:
                    builder.create_link(network, l, n)

                for l in links_to:
                    builder.create_link(network, n, l)

        return network

class With0Nodes(Template):

    def __init__(self, template, logger):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._logger = logger

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)
        for node in network.getNodes():
            if bayesianpy.network.is_variable_continuous(node.getVariables().get(0)):
                n = builder.create_discretised_variable(
                            network, self._template.get_network_factory().get_data(), node.getName() + "_0Node",
                            bins=[(jp.java.lang.Double.NEGATIVE_INFINITY, 0.5, "closed", "open"),
                                  (0.5, jp.java.lang.Double.POSITIVE_INFINITY, "closed", "closed")])

                builder.create_link(network, n, node)
        return network

from typing import Tuple
class MoveNode(Template):

    def __init__(self, template, target_node:str, delete_all_links=True, parents:List[str]=None, children:List[str]=None):
        self._template = template
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._target_node = target_node
        self._delete_all_links = delete_all_links
        self._parents = parents
        self._children = children


    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)

        node = bayesianpy.network.get_node(network, self._target_node)

        if node is None:
            raise ValueError("Node {} does not exist in network".format(self._target_node))

        if self._delete_all_links:
            builder.delete_links_from(network, node)
            builder.delete_links_to(network, node)

        for link in self._parents:
            builder.create_link(network, link, node)

        for link in self._children:
            builder.create_link(network, node, link)

        return network

class WithLatentNode(Template):

    def __init__(self, template, logger, latent_states=5, target_nodes=None, label:str=None, remove_target_node=False):
        super().__init__(discrete=template._discrete, continuous=template._continuous, label=label)
        self._template = template
        self._latent_states = latent_states
        self._logger = logger

        if not isinstance(target_nodes, list) and target_nodes is not None:
            target_nodes = [target_nodes]

        self._target_nodes = target_nodes
        self._remove_target_node = remove_target_node

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)

        cluster = builder.create_cluster_variable(network, self._latent_states)

        for node in bayesianpy.network.get_nodes(network):
            if node == cluster:
                continue
            builder.create_link(network, cluster, node)

        if self._target_nodes is not None:
            for target_node in self._target_nodes:
                target = builder.get_node(network, target_node)
                builder.delete_links_from(network, target)

                if self._remove_target_node:
                    bayesianpy.network.remove_node(network, self._target_nodes)

        return network


class WithEdges(Template):

    def __init__(self, template, logger, connections=[]):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._connections = connections
        self._logger = logger

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)
        for connection_route in self._connections:
            for i in range(0, len(connection_route)-1):
                try:
                    builder.create_link(network, connection_route[i], connection_route[i+1])
                except ValueError as e:
                    self._logger.warn(e)
                    continue

        return network

class WithFullyConnectedNodes(Template):

    def __init__(self, template, fully_connected_nodes=[]):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._fully_connected_nodes = fully_connected_nodes

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)
        for node in self._fully_connected_nodes:
            for child in network.getNodes():
                if bayesianpy.network.is_cluster_variable(child):
                    continue
                
                if child.getName() == node:
                    continue

                try:
                    builder.create_link(network, node, child)
                except ValueError:
                    continue

        return network

class WithMultivariateNodes(Template):

    def __init__(self, template, connections=[]):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._connections = connections

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)
        for connection_route in self._connections:
            for i in range(0, len(connection_route)):
                builder.create_link(network, connection_route[i], connection_route[i+1])

        return network


