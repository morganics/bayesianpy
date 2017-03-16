from bayesianpy.network import Builder as builder
import pandas as pd
import bayesianpy.network
from bayesianpy.jni import *
import bayesianpy.dask as dk


class Template:
    def __init__(self, discrete=pd.DataFrame(), continuous=pd.DataFrame()):
        self._discrete = discrete
        self._continuous = continuous

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        pass

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
                    states = self._discrete_states[d_name]
                else:
                    states = dk.compute(self._discrete[d_name].dropna().unique()).tolist()

                try:
                    c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                
                    builder.create_link(network, cluster, c)
                except BaseException as e:
                    self._logger.warn(e)


        return network

class NetworkWithoutEdges(Template):
    def __init__(self, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10,
                 discrete_states={}):
        super().__init__(discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._discrete_states = discrete_states

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = network_factory.create()
        #builder.create_cluster_variable(network, 5)

        if not self._continuous.empty:
            for c_name in self._continuous.columns:
                builder.create_continuous_variable(network, c_name)

        if not self._discrete.empty:
            for d_name in self._discrete.columns:
                if d_name in self._discrete_states:
                    states = self._discrete_states[d_name]
                else:
                    states = self._discrete[d_name].dropna().unique()

                builder.create_discrete_variable(network, self._discrete, d_name, states)

        return network

class DiscretisedMixtureNaiveBayes(Template):

    def __init__(self, logger, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10, bin_count=4,
                 binning_mode='EqualFrequencies'):
        super().__init__(discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._logger = logger
        self._bin_count = bin_count
        self._binning_mode = binning_mode

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = network_factory.create()
        cluster = builder.create_cluster_variable(network, self._latent_states)

        if not self._continuous.empty:
            for c_name in self._continuous.columns:
                c = builder.create_discretised_variable(network, self._continuous, c_name, bin_count=self._bin_count,
                                                        mode=self._binning_mode)

                builder.create_link(network, cluster, c)

        if not self._discrete.empty:
            for d_name in self._discrete.columns:
                states = self._discrete[d_name].dropna().unique()
                c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                builder.create_link(network, cluster, c)

        return network

class AutoStructure(Template):
    def __init__(self, template, data_store, logger):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._data_store = data_store
        self._logger = logger

    def learn(self, network_factory: bayesianpy.network.NetworkFactory):

        data_reader_command = self._data_store.create_data_reader_command()

        reader_options = bayesServer().data.ReaderOptions()
        network = self._template.create(network_factory)
        network.getLinks().clear()

        variable_references = list(bayesianpy.network.create_variable_references(network, self._data_store.get_dataframe()))
        evidence_reader_command = bayesServer().data.DefaultEvidenceReaderCommand(data_reader_command, jp.java.util.Arrays.asList(variable_references), reader_options)

        options = bayesServerStructure().PCStructuralLearningOptions()
        options.setMaximumConditional(2)
        self._logger.info("Learning structure from {} variables.".format(len(variable_references)))
        output = bayesServerStructure().PCStructuralLearning().learn(evidence_reader_command, jp.java.util.Arrays.asList(network.getNodes().toArray()),
                                                                 options)

        self._logger.info("Created {} links.".format(len(output.getLinkOutputs())))

        for link in output.getLinkOutputs():
            self._logger.debug("Link added from {} -> {}".format(
                              link.getLink().getFrom().getName(),
                              link.getLink().getTo().getName()))

        return output


    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = self._template.create(network_factory)

        for link in self.learn(network_factory).getLinkOutputs():
            try:
                builder.create_link(network, link.getLink().getFrom().getName(), link.getLink().getTo().getName())
            except ValueError:
                self._logger.warn("Could not add link from {} to {}".format(link.getLink().getFrom().getName(), link.getLink().getTo().getName()))

        return network

class WithDiscretisedVariables(Template):
    def __init__(self, template: Template, logger, discretised_variables=[], bins=[], mode='EqualFrequencies'):
        super().__init__(discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._discretised_variables = discretised_variables
        self._logger = logger
        self._bins = bins
        self._mode = mode

        if len(self._bins) != len(self._discretised_variables):
            raise ValueError("Bins and variables count should be the same")

    def create(self, network_factory: bayesianpy.network.NetworkFactory):
        network = network_factory.create()
        for i, var in enumerate(self._discretised_variables):
            node = builder.get_node(network, var)
            if node is None:
                raise ValueError("{} does not exist".format(var))

            links_from = [link.getFrom() for link in node.getLinks() if link.getFrom().getName() != var]
            links_to = [link.getTo() for link in node.getLinks() if link.getTo().getName() != var]

            network.getNodes().remove(node)

            n = builder.create_discretised_variable(network, network_factory.get_data(), var,
                                                    bin_count=self._bins[i], mode=self._mode)
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


