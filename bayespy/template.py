from bayespy.network import Builder as builder
import pandas as pd
import bayespy.network
from bayespy.jni import *

class Template:
    def __init__(self, network_factory, discrete=pd.DataFrame(), continuous=pd.DataFrame()):
        self._discrete = discrete
        self._continuous = continuous
        self._network_factory = network_factory

    def get_network_factory(self):
        return self._network_factory

    def create(self):
        pass

class MixtureNaiveBayes(Template):

    def __init__(self, network_factory, logger, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10, discrete_states={}):
        super().__init__(network_factory, discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._discrete_states = discrete_states
        self._logger = logger

    def create(self):
        network = self._network_factory.create()
        cluster = builder.create_cluster_variable(network, self._latent_states)

        if not self._continuous.empty:
            for c_name in self._continuous.columns:
                c = builder.create_continuous_variable(network, c_name)
                try:
                    builder.create_link(network, cluster, c)
                except ValueError as e:
                    self._logger.warn(e)

        if not self._discrete.empty:
            for d_name in self._discrete.columns:
                if d_name in self._discrete_states:
                    states = self._discrete_states[d_name]
                else:
                    states = self._discrete[d_name].dropna().unique()

                try:
                    c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                
                    builder.create_link(network, cluster, c)
                except BaseException as e:
                    self._logger.warn(e)


        return network

class NetworkWithoutEdges(Template):
    def __init__(self, network_factory, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10,
                 discrete_states={}):
        super().__init__(network_factory, discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._discrete_states = discrete_states

    def create(self):
        network = self._network_factory.create()
        #builder.create_cluster_variable(network, 5)

        if not self._continuous.empty:
            for c_name in self._continuous.columns:
                builder.create_continuous_variable(network, c_name)

        if not self._discrete.empty:
            for d_name in self._discrete.columns:
                if d_name in self._discrete_states:
                    states = self._discrete_states[d_name]
                else:
                    print(d_name)
                    states = self._discrete[d_name].dropna().unique()

                builder.create_discrete_variable(network, self._discrete, d_name, states)

        return network

class DiscretisedMixtureNaiveBayes(Template):

    def __init__(self, network_factory, logger, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10):
        super().__init__(network_factory, discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._logger = logger

    def create(self):
        network = self._network_factory.create()
        cluster = builder.create_cluster_variable(network, 5)

        if not self._continuous.empty:
            for c_name in self._continuous.columns:
                c = builder.create_discretised_variable(network, self._continuous, c_name)
                builder.create_link(network, cluster, c)

        if not self._discrete.empty:
            for d_name in self._discrete.columns:
                states = self._discrete[d_name].dropna().unique()
                c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                builder.create_link(network, cluster, c)

        return network

class AutoStructure(Template):
    def __init__(self, template, data_store, logger):
        super().__init__(template.get_network_factory(), discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._data_store = data_store
        self._logger = logger

    def learn(self):

        data_reader_command = self._data_store.create_data_reader_command()

        reader_options = bayesServer.data.ReaderOptions()
        network = self._template.create()
        network.getLinks().clear()

        variable_references = list(bayespy.network.create_variable_references(network, self._data_store.get_dataframe()))
        evidence_reader_command = bayesServer.data.DefaultEvidenceReaderCommand(data_reader_command, jp.java.util.Arrays.asList(variable_references), reader_options)

        self._logger.info("Learning structure from {} variables.".format(len(variable_references)))
        output = bayesServerStructure.PCStructuralLearning().learn(evidence_reader_command, jp.java.util.Arrays.asList(network.getNodes().toArray()),
                                                                 bayesServerStructure.PCStructuralLearningOptions())

        self._logger.info("Created {} links.".format(len(output.getLinkOutputs())))

        for link in output.getLinkOutputs():
            self._logger.debug("Link added from {} -> {}".format(
                              link.getLink().getFrom().getName(),
                              link.getLink().getTo().getName()))

        return output


    def create(self):
        network = self._template.create()
        network.getLinks().clear()

        for link in self.learn().getLinkOutputs():
            try:
                builder.create_link(network, link.getLink().getFrom().getName(), link.getLink().getTo().getName())
            except ValueError:
                self._logger.warn("Could not add link from {} to {}".format(link.getLink().getFrom().getName(), link.getLink().getTo().getName()))

        return network

class WithDiscretisedVariables(Template):
    def __init__(self, template: Template, logger, discretised_variables=[], bins=[], mode='EqualFrequencies'):
        super().__init__(template.get_network_factory(), discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._discretised_variables = discretised_variables
        self._logger = logger
        self._bins = bins
        self._mode = mode

        if len(self._bins) != len(self._discretised_variables):
            raise ValueError("Bins and variables count should be the same")

    def create(self):
        network = self._template.create()

        for i, var in enumerate(self._discretised_variables):
            node = builder.get_node(network, var)
            if node is None:
                raise ValueError("{} does not exist".format(var))

            links_from = [link.getFrom() for link in node.getLinks() if link.getFrom().getName() != var]
            links_to = [link.getTo() for link in node.getLinks() if link.getTo().getName() != var]

            network.getNodes().remove(node)

            n = builder.create_discretised_variable(network, self._template.get_network_factory().get_data(), var,
                                                    bin_count=self._bins[i], mode=self._mode)
            for l in links_from:
                builder.create_link(network, l, n)

            for l in links_to:
                builder.create_link(network, n, l)

        return network

class WithEdges(Template):

    def __init__(self, template, logger, connections=[]):
        super().__init__(template.get_network_factory(), discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._connections = connections
        self._logger = logger

    def create(self):

        network = self._template.create()
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
        super().__init__(template.get_network_factory(), discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._fully_connected_nodes = fully_connected_nodes

    def create(self):
        network = self._template.create()
        for node in self._fully_connected_nodes:
            for child in network.getNodes():
                if bayespy.network.is_cluster_variable(child):
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
        super().__init__(template.get_network_factory(), discrete=template._discrete, continuous=template._continuous)
        self._template = template
        self._connections = connections

    def create(self):
        network = self._template.create()
        for connection_route in self._connections:
            for i in range(0, len(connection_route)):
                builder.create_link(network, connection_route[i], connection_route[i+1])

        return network


