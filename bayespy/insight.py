import pandas as pd
from bayespy.network import Discrete
import bayespy.network
import numpy as np
from collections import Counter
import bayespy.data
import bayespy.jni
from bayespy.jni import jp


class _AutoInsight:

    def __init__(self, network, target, logger):
        self._network = network
        self._logger = logger
        self._target = target
        self._target_state = bayespy.network.get_state(network, target.variable, target.state)
        self._target = target
        (self._inf_engine, _, _) = bayespy.model.InferenceEngine(network).create(retract=False)

    def calculate(self, evidence=[], sort_by=['difference']):

        variables = jp.java.util.Arrays.asList(
            [v for v in self._network.getVariables() if v.getName() != self._target.variable])

        ai = bayespy.jni.bayesServerAnalysis.AutoInsight

        if len(evidence) > 0:
            e = bayespy.model.Evidence(self._network, self._inf_engine)
            evidence_obj = e.apply(evidence)
            auto_insight_output = ai.calculate(self._target_state, variables,
                                           bayespy.model.InferenceEngine.inference_factory,
                                           evidence_obj)
        else:
            auto_insight_output = ai.calculate(self._target_state, variables,
                                               bayespy.model.InferenceEngine.inference_factory)

        results = []
        for variable in auto_insight_output.getVariableOutputs():
            variable_name = variable.getVariable().getName()

            if variable_name == "Cluster":
                continue

            for state in variable.getStateOutputs():
                results.append({'variable': variable_name, 'state': state.getState().getName(),
                                'probability': state.getProbability().floatValue(),
                                'probability_given_target': state.getProbabilityGivenTarget().floatValue(),
                                'probability_target_given_this': np.nan if state.getProbabilityTargetGivenThis() is None else state.getProbabilityTargetGivenThis().floatValue(),
                                'difference': state.getDifference().floatValue(),
                                'lift': np.nan if state.getLift() is None else state.getLift().floatValue()})

        df = pd.DataFrame(results)
        return df.sort_values(by=sort_by, ascending=False).reset_index().drop('index', axis=1)


class AutoInsight:
    def __init__(self, template, target, logger, comparison_models=3):
        self._network_template = template
        self._logger = logger
        self._data_store = template.get_network_factory().get_datastore()
        self._model_cache = []
        self._comparison_model_count = comparison_models
        self._target = target

    def _create_models(self):
        if len(self._model_cache) > 0:
            return self._model_cache

        for i in range(self._comparison_model_count):
            network = self._network_template.create()
            model = bayespy.model.NetworkModel(network, self._data_store, self._logger)
            model.train()
            model.save(r"C:\Users\imorgan.admin\PycharmProjects\bayespy\examples\titanic\{}.bayes".format(i))
            self._model_cache.append(_AutoInsight(network, self._target, self._logger))

        return self._model_cache

    def get_most_common_tuples(self, combination_length=2, top=20):
        models = self._create_models()

        group = 0

        combinations = pd.DataFrame()
        for model in models:
            insight = model.calculate()
            reader = bayespy.data.DataFrameReader(insight)
            while reader.read():
                rows = [reader.to_dict()]
                evidence = [bayespy.network.Discrete(reader.variable, reader.state)]
                for i in range(combination_length-1):
                    sub_insight = model.calculate(evidence=evidence)
                    top_row = sub_insight.iloc[0]
                    evidence.append(bayespy.network.Discrete(top_row.variable, top_row.state))
                    d = top_row.to_dict()
                    d.update({'group': group})
                    rows.append(d)

                r = pd.DataFrame(rows)
                r['max_difference'] = r.difference.sum()
                r['evidence'] = ','.join([str(n) for n in evidence])
                combinations = combinations.append(r)

                group += 1

        return combinations.groupby(by=['evidence']).mean().sort_values(by=['max_difference'], ascending=False)\
            .reset_index().drop(['index', 'group'], axis=1).head(top)

    def get_descriptive_combinations(self, top=10):
        models = self._create_models()

        combinations = pd.DataFrame()
        for i, model in enumerate(models):
            rows = []
            evidence = []
            for j in range(10):
                step = model.calculate(evidence=evidence)
                row = step.iloc[0]
                evidence.append(bayespy.network.Discrete(row.variable, row.state))
                d = row.to_dict()
                d.update({'group': i})
                rows.append(d)
                if row.difference < 0.05:
                    break

            r = pd.DataFrame(rows)
            r['max_difference'] = r.difference.sum()
            r['evidence'] = ','.join([str(n) for n in evidence])
            combinations = combinations.append(r)

        return combinations.sort_values(by=['max_difference']).reset_index()

    def get_exclusive_states(self, top=10):
        return self.get_insightful_states(using='lift', top=top)

    def get_insightful_states(self, using='difference', top=10):
        if using not in ['lift', 'difference']:
            raise ValueError("Expecting either lift or difference in the using parameter. Difference favours probability"
                             " changes with a higher likelihood of occurring, while lift favours relative changes in probability"
                             " without taking in to account the likelihood that they will occur.")

        models = self._create_models()
        rows = pd.DataFrame()
        for model in models:
            rows = rows.append(model.calculate(), ignore_index=True)

        return rows.groupby(by=['variable', 'state']).mean().sort_values(by=[using], ascending=False).head(
            top).reset_index()





            #
            # selected_items = pd.DataFrame(selected_items)
            # print(selected_items)

    # def get_combinations(self, target):
    #
    #     models = self._create_models()
    #
    #     for model in
    #
    #         evidence = []
    #         selected_items = []
    #
    #         for i in range(10):
    #             step = a.calculate(evidence=evidence)
    #             row = step.iloc[x]
    #             selected_items.append(row.to_dict())
    #             evidence.append(bayespy.network.Discrete(row.variable, row.state))
    #             if row.difference < 0.05:
    #                 break
    #
    #         selected_items = pd.DataFrame(selected_items)
    #         print(selected_items)



class AutoInsight1:
    def __init__(self, network_factory, logger, continuous=[], discrete=[]):
        self._continuous = continuous
        self._discrete = discrete
        self._factory = network_factory
        self._logger = logger

    def evidence_query(self, model=None, base_evidence=None, new_evidence=None):
        """
        deprecated, use query_model_with_evidence instead.
        :param model:
        :param base_evidence:
        :param new_evidence:
        :return:
        """
        return self.query_model_with_evidence(model=model, base_evidence=base_evidence, new_evidence=new_evidence)

    def query_model_with_evidence(self, model=None, base_evidence=None, new_evidence=None):
        """
        Get the difference query between base_evidence and new_evidence
        :param model: The trained model, or None
        :param base_evidence: the base_evidence of the model
        :param new_evidence: the new_evidence applied to the model
        :return: a dataframe with variable, state, base probability, new probability and difference between queries
        """
        if model is None:
            model = self._get_trained_model()

        inference = model.inference()
        if base_evidence is not None:
            model.evidence(inference).apply(base_evidence)

        output = model.create_query(inference).execute()

        output.discrete.rename(columns={"value": "base_probability"}, inplace=True)
        output.continuous.rename(columns={"mean": "base_mean", "variance": "base_variance"}, inplace=True)

        if new_evidence is None:
            return (output.discrete, output.continuous)

        model.evidence(inference).apply(new_evidence)

        output_1 = model.create_query(inference).execute()

        c_2 = pd.merge(output_1.continuous, output.continuous, on=['variable'])
        o_2 = pd.merge(output_1.discrete, output.discrete, on=['variable', 'state'])
        o_2['difference'] = o_2['value'] - o_2['base_probability']

        o_2['variable_state'] = o_2.variable.str.cat(others=o_2.state, sep=bayespy.network.STATE_DELIMITER)
        return (o_2, c_2)

    def _resolve_cluster_index(self, df, variable, state):
        """
        Get a consistent cluster index across model builds, as a discrete Cluster_ node may not consistently
        map to the underlying continuous child node
        :param df: the dataframe (from a query to the model)
        :param variable: the current discrete variable
        :param state: the current discrete state
        :return: an integer relating to the appropriate cluster, to be used internally
        """
        means = df[df.variable == variable][['state', 'mean']].copy()
        means = means.sort_values(by='mean').reset_index()
        return means[means.state == state].index[0]

    def _resolve_cluster_from_index(self, df, variable, index):
        """
        Map from the internal cluster index to the external variable name, given a dataframe.
        :param df:
        :param variable:
        :param index:
        :return: the row relating to the external variable
        """
        means = df[df.variable == variable].copy()
        means = means.sort_values(by='mean').reset_index()
        return means.ix[int(index)]

    def _get_trained_model(self, network):
        self._logger.debug("Training model")
        return self._factory.create_trained_model(network, self._discrete.index.tolist())

    def create_model_cache(self, target, times=1):
        return [m for f, m in self._build_and_query_network(target, times=times)]

    def query_exclusive_states(self, target, models=None, times=1, top=10):
        """
        Get variables where a target has almost guaranteed likelihood, even if it has a small difference
        :param target: the target state (a Discrete instance)
        :param times: the number of times a model should be trained and queried
        :param top: the top N results
        :return: a list of (string) variables
        """

        features = self._build_and_query_network(target, models=models, times=times)
        cc = Counter()
        models = []
        for df, model in features:
            dfr = bayespy.data.DataFrameReader(
                df[(df.base_probability < 0.008) & (df.difference > 0.005) & (df.difference < 1)])
            while dfr.read():
                if dfr['variable'] == target.variable or dfr['variable'] == 'Cluster':
                    continue

                if bayespy.network.is_cluster_variable(dfr['variable']):
                    ix = self._resolve_cluster_index(df, dfr['variable'], dfr['state'])
                    cc["{}{}{}".format(dfr['variable'], bayespy.network.STATE_DELIMITER, ix)] += dfr['difference']

                else:
                    cc[dfr['variable_state']] += dfr['difference']

            models.append(model)

        results = self._get_variable_frequency(features, cc, top)

        return results

    # def query_top_discriminative_variables(self, target, models=None, times=1, top=10):
    #     generator = self.query_top_variable_combinations_as_df(target, models=models, times=times, top=top)
    #     for df, model in generator:
    #         for row in bayespy.data.DataFrameReader(df):
    #             self._resolve_cluster_index(df, row['variable']

    def query_bivariate_combinations(self, target, models=None, times=1, top=10):
        """
        Get the most significant bivariate (target + one other) combinations,
        where the variables with the highest difference values are returned
        :param target: the target variable (a Discrete instance)
        :param times: the number of times a model should be trained and queried
        :param top: top N results
        :return: a list of (string) variables
        """
        features = self._build_and_query_network(target, models=models, times=times)

        cc = Counter()
        models = []
        for df, model in features:
            dfr = bayespy.data.DataFrameReader(df)
            while dfr.read():
                if dfr['variable'] == target.variable or dfr['variable'] == 'Cluster':
                    continue

                if bayespy.network.is_cluster_variable(dfr['variable']):
                    ix = self._resolve_cluster_index(df, dfr['variable'], dfr['state'])
                    cc["{}{}{}".format(dfr['variable'], bayespy.network.STATE_DELIMITER, ix)] += dfr['difference']

                else:
                    cc[dfr['variable_state']] += dfr['difference']

            models.append(model)

        results = self._get_variable_frequency(features, cc, top)

        return results

    def _get_mean_value_across_models(self, models, discrete, key):
        values = []
        for model in models:
            values.append(float(model[(model.variable == discrete.variable) & (model.state == discrete.state)][key]))

        return np.mean(values)

    def _get_variable_frequency(self, query, counter, top):
        most_common = counter.most_common(top)
        mc = []
        models = [m[0] for m in query]
        for (v, d) in most_common:
            v_ = bayespy.network.Discrete.fromstring(v)
            if bayespy.network.is_cluster_variable(v_.variable):
                av = []
                va = []
                for df, model in query:
                    c = self._resolve_cluster_from_index(df, v_.variable, v_.state)
                    av.append(c['mean'])
                    va.append(c['variance'])

                #target_probability = self._get_mean_value_across_models(models, v_, 'target_probability')
                mc.append((bayespy.network.Discrete(v_.variable, c['state']).tostring(),
                           {'mean': np.mean(av), 'variance': np.mean(va), 'sum_difference': d, 'target_probability': 0}))
            else:
                base_probability = self._get_mean_value_across_models(models, v_, 'base_probability')
                target_probability = self._get_mean_value_across_models(models, v_, 'target_probability')
                mc.append((v, {'base_probability': base_probability, 'target_probability': target_probability,
                               'sum_difference': d}))

        return mc

    def _build_and_query_network(self, target, times=1, models=None):
        """
        Builds the network, trains it and queries it to get difference between all other states in target variable and the target state.
        :param target: the target (a Discrete instance)
        :param times: the number of times to train/ query
        :return: a list of dataframes (len(return) == times)
        """
        i = 0

        if models is not None:
            self._logger.info(
                "The number of cached models sets the number of iterations, equal to {}".format(len(models)))

        if not isinstance(target, Discrete):
            raise ValueError("target should be of type discretenode")

        (network, network_builder) = self._factory.create_network()
        network_builder.build_naive_network_with_latent_parents(discrete=self._discrete,
                                                                continuous=self._continuous, latent_states=10)

        if not bayespy.network.is_variable_discrete(bayespy.network.get_variable(network, target.variable)):
            raise ValueError("Target variable '{}' is not discrete.".format(target.variable))

        target_alt = list(bayespy.network.get_other_states_from_variable(network, target))
        self._logger.debug("Finished building network.")


        features = []
        while len(features) < times:
            if models is not None:
                model = models[len(features)]
                self._logger.debug("Pulled model from cache.")
            else:
                model = self._get_trained_model(network.copy())
                self._logger.debug("Trained model")

            t = [target.tostring()]

            # get a list of the most different nodes between all other and the target node.
            (discrete_features, continuous_features) = self.query_model_with_evidence(model=model,
                                                                                      base_evidence=target_alt,
                                                                                      new_evidence=t)

            # get the continuous_variable_name if it's a parent of a continuous node.
            discrete_features['continuous_variable_name'] = discrete_features.variable.apply(
                lambda x: x.replace("Cluster_", "") if "Cluster_" in x else np.nan)

            # get the mean and variance for each of the continuous nodes and which are assumed to be children of the discrete nodes.
            discrete_features['mean'] = 0.0
            discrete_features['variance'] = 0.0
            discrete_features['target_probability'] = 0.0

            dfr = bayespy.data.DataFrameReader(discrete_features)
            while dfr.read():
                if dfr['variable'] == target.variable or dfr['variable'] == 'Cluster':
                    continue

                (ds, cs) = self.query_model_with_evidence(model=model, new_evidence=[dfr['variable_state']])
                if bayespy.network.is_cluster_variable(dfr['variable']):
                    discrete_features.set_value(dfr.get_index(), 'mean',
                                                cs[cs.variable == dfr['continuous_variable_name']]['mean'])
                    discrete_features.set_value(dfr.get_index(), 'variance',
                                                cs[cs.variable == dfr['continuous_variable_name']]['variance'])

                # gets the target value given the particular evidence set on the discrete variable
                discrete_features.set_value(dfr.get_index(), 'target_probability', ds[(ds.variable == target.variable) & (ds.state == target.state)]['value'])

            discrete_features.sort_values(by=['difference'], inplace=True, ascending=False)

            features.append((discrete_features, model))

        return features

    @staticmethod
    def _get_row(df, target):
        for i in range(0, 10):
            if df.ix[i].variable == "Cluster" or df.ix[i].variable == target.variable:
                continue

            return df.ix[i]

    def query_top_variable_combinations_as_df(self, target, models=None, times=5, top=3):
        results, models = self.query_top_variable_combinations(target, models=models, times=times, top=top)

        for i,result in enumerate(results):

            rows = []
            for i, item in enumerate(result['evidence']):
                row = {'mean': np.nan, 'variance': np.nan, 'variable': '', 'probability': 0.0, 'state': '',
                       'max_p': 0.0,
                       'difference': 0.0}
                d = bayespy.network.Discrete.fromstring(item)
                c_name = d.variable.replace("Cluster_", "")
                if c_name != d.variable:
                    for cv in result['continuous_variables']:
                        if cv['variable'] == c_name:
                            row['mean'] = cv['mean']
                            row['variance'] = cv['variance']

                row['difference'] = result['difference'][i]
                row['variable'] = d.variable
                row['state'] = d.state
                row['probability'] = result['probability'][i]
                row['max_p'] = result['max_probability']
                rows.append(row)

            yield pd.DataFrame(rows), models[i]

    def query_top_variable_combinations(self, target, models=None, times=5, top=3):
        (network, network_builder) = self._factory.create_network()

        if not isinstance(target, Discrete):
            raise ValueError("target should be of type discretenode")

        network_builder.build_naive_network_with_latent_parents(discrete=self._discrete,
                                                                continuous=self._continuous, latent_states=10)

        target_alt = list(bayespy.network.get_other_states_from_variable(network, target))
        self._logger.debug("Finished building network.")
        t = [target.tostring()]

        if models is not None:
            times = len(models)
            if top > len(models):
                top = len(models)

        combinations = []

        while len(combinations) < times:
            if models is not None:
                #print(len(combinations))
                model = models[len(combinations)]
                self._logger.debug("Picked up model from cache")
            else:
                model = self._get_trained_model(network)
                self._logger.debug("Trained model")

            base_evidence = []
            difference = []
            target_prob_given_evidence = []
            recent_variable = None
            continuous_variables = []
            for i in range(top):

                (discrete_features, continuous_features) = self.evidence_query(model=model,
                                                                               base_evidence=base_evidence + target_alt,
                                                                               new_evidence=base_evidence + t)

                discrete_features.sort_values(by='difference', inplace=True, ascending=False)
                discrete_features.reset_index(inplace=True)

                # get the target probability given evidence applied to the newly found variable
                (dsf, csf) = self.evidence_query(model=model, new_evidence=base_evidence)

                if recent_variable is not None and bayespy.network.is_cluster_variable(recent_variable.variable):
                    c_v_name = recent_variable.variable.replace("Cluster_", "")
                    continuous_variables.append({'mean': float(csf[c_v_name == csf.variable]['mean']),
                                                 'variance': float(csf[c_v_name == csf.variable]['variance']),
                                                 'variable': c_v_name})

                target_given_evidence = dsf[dsf.variable == target.variable]

                # print(target_given_evidence)
                target_prob_given_evidence.append(float(target_given_evidence[dsf.state == str(target.state)].value))
                row = self._get_row(discrete_features, target)
                recent_variable = Discrete(row.variable, row.state)

                difference.append(row.difference)
                base_evidence.append(recent_variable.tostring())

            combinations.append(
                {'evidence': base_evidence, 'difference': difference, 'probability': target_prob_given_evidence,
                 'max_probability': np.max(target_prob_given_evidence), 'continuous_variables': continuous_variables,
                 'model': model})

        combinations = sorted(combinations, key=lambda x: x['max_probability'], reverse=True)

        models = [combo.pop('model') for combo in combinations]

        return combinations, models

    def query_variable_combinations(self, target, conditioned=3, total_iterations_limit=10, diff_convergence_dp=4):

        (network, network_builder) = self._factory.create_network()

        if not isinstance(target, Discrete):
            raise ValueError("target should be of type discretenode")

        network_builder.build_naive_network_with_latent_parents(discrete=self._discrete,
                                                                continuous=self._continuous, latent_states=10)

        target_alt = list(bayespy.network.get_other_states_from_variable(network, target))
        self._logger.debug("Finished building network.")

        results = []
        total_over_limit = 0
        total_iterations = 0
        while total_over_limit <= conditioned:
            model = self._get_trained_model(network)
            self._logger.debug("Trained model")
            t = [target.tostring()]

            base_evidence = []
            prev_target_prob = np.nan
            curr_target_prob = np.nan
            difference = []
            while round(prev_target_prob, diff_convergence_dp) != round(curr_target_prob, diff_convergence_dp):
                prev_target_prob = curr_target_prob
                (discrete_features, continuous_features) = self.evidence_query(model=model,
                                                                               base_evidence=base_evidence + target_alt,
                                                                               new_evidence=base_evidence + t)

                discrete_features.sort_values(by='difference', inplace=True, ascending=False)
                discrete_features.reset_index(inplace=True)

                (dsf, csf) = self.evidence_query(model=model, new_evidence=base_evidence)

                target_given_evidence = dsf[dsf.variable == target.variable]
                # print(target_given_evidence)
                curr_target_prob = float(target_given_evidence[dsf.state == str(target.state)].value)
                self._logger.debug("Evidence: {0}".format(base_evidence))
                self._logger.debug(target_given_evidence)
                row = self._get_row(discrete_features, target)
                difference.append(row.difference)
                evi = Discrete(row.variable, row.state).tostring()
                base_evidence.append(evi)

            if prev_target_prob > 90:
                total_over_limit += 1

            total_iterations += 1

            results.append(
                {'evidence': base_evidence, 'difference': difference, 'probability': prev_target_prob, 'model': model})

            if total_iterations >= total_iterations_limit:
                break

            self._logger.debug("Iteration count: {0}".format(total_iterations))

        return results

    def rationalise(self, results, num=20):
        from collections import Counter

        top_results = Counter()
        for d in results:
            for j, _ in enumerate(d['evidence']):
                top_results[d['evidence'][j]] += (d['difference'][j] * d['probability'])
                # print(d['evidence'][j])
        return top_results.most_common(num)
