from bayespy.network import Builder as builder
import pandas as pd

class Template:
    def __init__(self, network_factory, discrete=pd.DataFrame(), continuous=pd.DataFrame()):
        self._discrete = discrete
        self._continuous = continuous
        self._network_factory = network_factory

class MixtureNaiveBayes(Template):

    def __init__(self, network_factory, discrete=pd.DataFrame(), continuous=pd.DataFrame(), latent_states=10, discrete_states={}):
        super().__init__(network_factory, discrete=discrete, continuous=continuous)
        self._latent_states = latent_states
        self._discrete_states = discrete_states

    def build(self):
        network = self._network_factory.create()
        cluster = builder.create_cluster_variable(network, 5)

        if not self._continuous.empty:
            for c_name in self._continuous.columns:
                c = builder.create_continuous_variable(network, c_name)
                builder.create_link(network, cluster, c)

        if not self._discrete.empty:
            for d_name in self._discrete.columns:
                if d_name in self._discrete_states:
                    states = self._discrete_states[d_name]
                else:
                    states = self._discrete[d_name].dropna().unique()

                c = builder.create_discrete_variable(network, self._discrete, d_name, states)
                builder.create_link(network, cluster, c)

        return network