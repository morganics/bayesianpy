import pandas as pd
from bayespy.network import Discrete
from bayespy.network import state
import numpy as np

class AutoInsight:

    def __init__(self, network_factory, continuous=[], discrete=[]):
        #self._model = model
        self._continuous = continuous
        self._discrete = discrete
        self._factory = network_factory

    @staticmethod
    def _get_row(df, target):
        for i in range(0,10):
            if df.ix[i].variable == "Cluster" or df.ix[i].variable == target.variable:
                continue

            #if random.random() <= 0.8:
            return df.ix[i]

    def evidence_query(self, model=None, base_evidence=None, new_evidence=None):
        #self._inference = self._factory.createInferenceEngine(self._network);
        if model is None:
            model = self._get_trained_model()

        inference = model.inference()
        if base_evidence is not None:
            #self._set_evidence(inference, variables, base_evidence)v
            model.evidence(inference).apply(base_evidence)

        output = model.create_query(inference).execute()

        output.discrete.rename(columns={"value": "base_probability"}, inplace=True)
        output.continuous.rename(columns={"mean" : "base_mean", "variance": "base_variance"}, inplace=True)

        if new_evidence is None:
            return (output.discrete, output.continuous)

        model.evidence(inference).apply(new_evidence)

        output_1 = model.create_query(inference).execute()

        c_2 = pd.merge(output_1.continuous, output.continuous, on=['variable'])
        o_2 = pd.merge(output_1.discrete, output.discrete, on=['variable', 'state'])
        o_2['difference'] = o_2['value'] -  o_2['base_probability']

        return (o_2, c_2)

    def _get_trained_model(self, network):
        return self._factory.create_trained_model(network, self._discrete.index.tolist())

    def query_exclusive_states(self, target):
        (network, network_builder) = self._factory.create_network()


        network_builder.build_naive_network_with_latent_parents(discrete=self._discrete,
                                                            continuous=self._continuous, latent_states=10)
        other = list(self._get_other_states(network, target))

        model = self._get_trained_model(network)

        (discrete_features, continuous_features) = self.evidence_query(model=model,
                                                                   base_evidence=other,
                                                                   new_evidence=[target.tostring()])

        return discrete_features[(np.isclose(discrete_features.value, 0)) | (discrete_features.base_probability.round(2) == 0.00)]

    def _get_other_states(self, network, target):
        target_ = network.getVariables().get(target.variable)
        for st in target_.getStates():
            if st.getName() == target.state:
                continue

            yield state(target.variable, st.getName())


    def query(self, target, conditioned=3, total_iterations_limit = 10):

        (network, network_builder) = self._factory.create_network()

        if not isinstance(target, Discrete):
            raise ValueError("target should be of type discretenode")

        network_builder.build_naive_network_with_latent_parents(discrete=self._discrete,
            continuous=self._continuous, latent_states=10)

        target_alt = list(self._get_other_states(network, target))

        results = []
        total_over_limit = 0
        total_iterations = 0
        while total_over_limit <= conditioned:
            model = self._get_trained_model(network)

            t = [target.tostring()]

            base_evidence = []
            prev_target_prob = np.nan
            curr_target_prob = np.nan
            difference = []
            while round(prev_target_prob, 5) != round(curr_target_prob, 5):
                prev_target_prob = curr_target_prob
                (discrete_features, continuous_features) = self.evidence_query(model=model, base_evidence=base_evidence + target_alt, new_evidence=base_evidence + t)

                discrete_features.sort_values(by='difference', inplace=True, ascending=False)
                discrete_features.reset_index(inplace=True)

                (dsf, csf) = self.evidence_query(model=model, new_evidence=base_evidence)

                target_given_evidence = dsf[dsf.variable == target.variable]
                curr_target_prob = float(target_given_evidence[dsf.state == target.state].value)
                print("Evidence: {0}".format(base_evidence))
                print(target_given_evidence)
                print(prev_target_prob, curr_target_prob)
                print("---")

                row = self._get_row(discrete_features, target)
                difference.append(row.difference)
                evi = Discrete(row.variable, row.state).tostring()
                base_evidence.append(evi)

            if prev_target_prob > 90:
                total_over_limit += 1

            total_iterations += 1

            results.append({ 'evidence': base_evidence, 'difference': difference, 'probability': prev_target_prob, 'model': model })

            if total_iterations >= total_iterations_limit:
                break

        return results

    def rationalise(self, results, num=20):
        from collections import Counter

        top_results = Counter()
        for d in results:
            for j,_ in enumerate(d['evidence']):
                top_results[d['evidence'][j]] += (d['difference'][j] * d['probability'])
                #print(d['evidence'][j])
        return top_results.most_common(num)