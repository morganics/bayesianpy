# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:31:18 2016

@author: imorgan.admin
"""

from bayesianpy.jni import bayesServerInference
from bayesianpy.decorators import deprecated

import bayesianpy.network
import pandas as pd
import bayesianpy.jni
from bayesianpy.jni import bayesServer
from bayesianpy.jni import bayesServerStatistics
from bayesianpy.jni import jp
import numpy as np
import logging
import multiprocess.context as ctx
import pathos.multiprocessing as mp
import itertools
import bayesianpy.reader
from typing import List, Dict, Tuple
import dask.dataframe as dd
import math


class QueryBase:
    def setup(self, network, inference_engine, query_options) -> None:
        pass

    def results(self, inference_engine, query_output) -> dict:
        pass

    def reset(self):
        pass


class Distribution:

    def __init__(self, head_variables: List[str], tail_variables: List[str], states: List[str]):
        self._tail_variables = tail_variables
        self._states = states
        self._head_variables = head_variables
        self._key = self.pretty_print()
        self._mean_values = []
        self._variance_value = np.nan
        self._mean_value = np.nan
        self._covariance = np.zeros((len(self._head_variables), len(self._head_variables)))

    def set_mean_variance(self, mean: float, variance: float):
        self._mean_value = mean
        self._variance_value = variance

    def append_mean(self, mean: float):
        self._mean_values.append(mean)

    def set_covariance_value(self, i: int, j: int, covariance: float):
        self._covariance[i, j] = covariance

    def get_cov_by_variable(self, variable_i: str, variable_j: str) -> np.array:
        i = self._head_variables.index(variable_i)
        j = self._head_variables.index(variable_j)
        c = self._covariance
        return np.array([[c[i, i], c[i, j]], [c[j, i], c[j, j]]], np.float64)

    def get_mean_by_variable(self, variable_i, variable_j) -> float:
        i = self._head_variables.index(variable_i)
        j = self._head_variables.index(variable_j)
        return (self._mean_values[i], self._mean_values[j])

    def get_mean(self) -> float:
        return self._mean_value

    def get_variance(self) -> float:
        return self._variance_value

    def get_std(self) -> float:
        return math.sqrt(self.get_variance())

    def get_covariance(self) -> np.array:
        return self._covariance

    def get_tail_variables(self) -> List[str]:
        return self._tail_variables

    def get_states(self) -> List[str]:
        return self._states

    def get_tail(self):
        for i, v in enumerate(self._tail_variables):
            yield (v, self._states[i])

    def pretty_print(self) -> str:
        return "P({} | {})".format(
            ", ".join([v for v in self._head_variables]),
            ", ".join("{}={}".format(v, self._states[i]) for i, v in enumerate(self._tail_variables))
        )

    def pretty_print_tail(self) -> str:
        return ", ".join("{}={}".format(v, self._states[i]) for i, v in enumerate(self._tail_variables))

    def is_covariant(self) -> bool:
        return len(self._head_variables) > 1

    def key(self) -> tuple:
        return self._key

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return self._key == other.key()

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


class QueryConditionalJointProbability(QueryBase):
    def __init__(self, head_variables: List[str], tail_variables: List[str]):
        self._head_variables = head_variables
        self._tail_variables = tail_variables
        self._discrete_variables = []
        self._is_discrete_head = False

    def get_head_variables(self):
        return self._head_variables

    def get_tail_variables(self):
        return self._tail_variables

    def setup(self, network, inference_engine, query_options):
        contexts = []
        for h in self._head_variables + self._tail_variables:
            v = bayesianpy.network.get_variable(network, h)

            if bayesianpy.network.is_variable_discrete(v):
                if h in self._head_variables:
                    # raise ValueError("Bayespy only supports discrete tail variables (BayesServer is fine with it though!)")
                    self._is_discrete_head = True

                self._discrete_variables.append(v.getName())
            else:
                if h in self._tail_variables:
                    raise ValueError(
                        "Bayespy only supports continuous head variables (BayesServer is fine with it though!)")

            contexts.append(bayesServer().VariableContext(v, bayesServer().HeadTail.HEAD if h in self._head_variables
            else bayesServer().HeadTail.TAIL))

        self._network = network
        if self._is_discrete_head:
            self._distribution = bayesServer().Table(contexts)
        else:
            self._distribution = bayesServer().CLGaussian(contexts)

        self._query_distribution = bayesServerInference().QueryDistribution(self._distribution)
        inference_engine.getQueryDistributions().add(self._query_distribution)

    def results(self, inference_engine, query_output):
        results = {}

        def state_generator(variables):
            for v in variables:
                tv = bayesianpy.network.get_variable(self._network, v)
                yield [state for state in tv.getStates()]

        if self._is_discrete_head:
            rows = []
            query_variables = self._head_variables + self._tail_variables
            # creates dataframe of combinations + probability.
            for state_combinations in itertools.product(*state_generator(query_variables)):
                row = {}
                state_array = jp.JArray(state_combinations[0].getClass())(len(state_combinations))
                for i, state in enumerate(state_combinations):
                    state_array[i] = state

                for i, v in enumerate(state_combinations):
                    row.update({query_variables[i]: state_array[i].getName()})

                row.update({'probability': self._distribution.get(state_array)})
                rows.append(row)
            return rows
        else:
            if len(self._head_variables) == 2 and len(self._tail_variables) == 0:
                h0 = bayesianpy.network.get_variable(self._network, self._head_variables[0])
                h1 = bayesianpy.network.get_variable(self._network, self._head_variables[1])
                results.update({
                    "{}_{}_covariance".format(h0.getName(), h1.getName()): self._distribution.getCovariance(h0, h1),
                    "{}_mean".format(h0.getName()): self._distribution.getMean(h0),
                    "{}_mean".format(h1.getName()): self._distribution.getMean(h1)
                })
            else:
                for state_combinations in itertools.product(*state_generator(self._discrete_variables)):

                    state_array = jp.JArray(state_combinations[0].getClass())(len(state_combinations))
                    for i, state in enumerate(state_combinations):
                        state_array[i] = state

                        dist = Distribution(self._head_variables, self._tail_variables,
                                            [state.getName() for state in state_combinations])
                        for i, h in enumerate(self._head_variables):
                            v = bayesianpy.network.get_variable(self._network, h)
                            mean = self._distribution.getMean(v, state_array)
                            if dist.is_covariant():
                                dist.append_mean(mean)
                                for j, h1 in enumerate(self._head_variables):
                                    v1 = bayesianpy.network.get_variable(self._network, h1)
                                    cov = self._distribution.getCovariance(v, v1, state_array)
                                    dist.set_covariance_value(i, j, cov)
                            else:
                                dist.set_mean_variance(mean, self._distribution.getVariance(v, state_array))

                    results.update({dist.key(): dist})
        return results


class QueryJointProbability(QueryConditionalJointProbability):
    def __init__(self, head_variables: List[str]):
        super().__init__(head_variables, [])


@deprecated("Use 'QueryConditionalJointProbability' or 'QueryJointProbability' instead.")
class QueryMixtureOfGaussians(QueryConditionalJointProbability):
    def __init__(self, head_variables: List[str], tail_variables: List[str]):
        super().__init__(head_variables, tail_variables)


class QueryStatistics(QueryBase):
    def __init__(self, calc_loglikelihood=True, calc_conflict=False, loglikelihood_column='loglikelihood',
                 conflict_column='conflict'):
        self._calc_loglikelihood = calc_loglikelihood
        self._calc_conflict = calc_conflict
        self._loglikelihood_column = loglikelihood_column
        self._conflict_column = conflict_column

    def setup(self, network, inference_engine, query_options):
        query_options.setLogLikelihood(self._calc_loglikelihood)
        query_options.setConflict(self._calc_conflict)

    def results(self, inference_engine, query_output):
        result = {}
        if self._calc_loglikelihood:
            ll = query_output.getLogLikelihood()
            value = ll.floatValue() if ll is not None else np.nan
            result.update({self._loglikelihood_column: value})

        if self._calc_conflict:
            result.update({self._conflict_column: query_output.getConflict().floatValue()})

        return result


# seems like a better name than QueryStatistics, so just having this here.
class QueryModelStatistics(QueryStatistics):
    def __init__(self, calc_loglikelihood=True, calc_conflict=False, loglikelihood_column='loglikelihood',
                 conflict_column='conflict'):
        super().__init__(calc_loglikelihood, calc_conflict, loglikelihood_column, conflict_column)


class QueryFactory:
    def __init__(self, class_type, *args, **kwargs):
        assert isinstance(class_type, type), "Needs to be a type"

        self._class_type = class_type
        self._args = args
        self._kwargs = kwargs

    def create(self):
        return self._class_type(*self._args, **self._kwargs)


class QueryMostLikelyState(QueryBase):
    def __init__(self, target_variable_name, output_dtype="object", suffix="_maxlikelihood"):
        self._target_variable_name = target_variable_name
        self._distribution = None
        self._output_dtype = output_dtype
        self._suffix = suffix
        self._variable = None

    def setup(self, network, inference_engine, query_options):
        distribution = None

        self._variable = bayesianpy.network.get_variable(network, self._target_variable_name)

        if bayesianpy.network.is_variable_discrete(self._variable):
            distribution = bayesServer().Table(self._variable)

        if distribution is None:
            raise ValueError("{} needs to be discrete in QueryMostLikelyState".format(self._target_variable_name))

        query_options.setQueryEvidenceMode(bayesServerInference().QueryEvidenceMode.RETRACT_QUERY_EVIDENCE)
        qd = bayesServerInference().QueryDistribution(distribution)

        self._distribution = distribution
        inference_engine.getQueryDistributions().add(qd)

    def reset(self):
        self._distribution = None
        self._variable = None

    def results(self, inference_engine, query_output):
        states = {}

        for state in self._variable.getStates():
            states.update({state.getName(): self._distribution.get([state])})

        # get the most likely state
        max_state = max(states.keys(), key=(lambda key: states[key]))
        max_state_name = bayesianpy.data.DataFrame.cast2(self._output_dtype, max_state)

        return {self._target_variable_name + self._suffix: max_state_name}


class QueryStateProbability(QueryMostLikelyState):

    def __init__(self, target_variable_name, variable_state_separator=bayesianpy.network.STATE_DELIMITER,
                 suffix="_probability", target_state_name: str = None):
        super().__init__(target_variable_name=target_variable_name, output_dtype="float64", suffix=suffix)
        self._variable_state_separator = variable_state_separator
        self._target_state_name = target_state_name

    def setup(self, network, inference_engine, query_options):
        super().setup(network, inference_engine, query_options)

    def results(self, inference_engine, query_output):
        states = {}
        for state in self._variable.getStates():
            if self._target_state_name is not None and state.getName() != self._target_state_name:
                continue

            p = self._distribution.get([state])
            if self._target_state_name is not None:
                states.update({self._target_variable_name + self._suffix: p})
            else:
                states.update({self._target_variable_name + self._variable_state_separator + state.getName()
                               + self._suffix: p})

        if len(states) == 0:
            raise ValueError("QueryStateProbability: the target state name did not match any variables")

        return states


class QueryLogLikelihood(QueryBase):
    def __init__(self, variable_names, column_name: str = '_loglikelihood', append_variable_names=True):
        if isinstance(variable_names, str):
            variable_names = [variable_names]

        if len(variable_names) == 0:
            raise ValueError("QueryLogLikelihood: Requires a non-empty list of variables for creating a distribution")

        if len(set(variable_names)) != len(variable_names):
            raise ValueError("QueryLogLikelihood: There are duplicate variable names in the query: {}".format(
                ", ".join(variable_names)))

        self._variable_names = variable_names
        self._distribution = None
        self._query_distribution = None
        self._append_variable_names = append_variable_names
        self._column_name = column_name

    def setup(self, network, inference_engine, query_options):
        variables = [bayesianpy.network.get_variable(network, n) for n in self._variable_names]

        if len(variables) == 0:
            raise ValueError("QueryLogLikelihood: Requires a non-empty list for creating a distribution")

        if len(variables) == 1:
            self._distribution = bayesServer().CLGaussian(variables[0])
        else:
            self._distribution = bayesServer().CLGaussian(variables)

        query_options.setQueryEvidenceMode(bayesServerInference().QueryEvidenceMode.RETRACT_QUERY_EVIDENCE)
        qd = bayesServerInference().QueryDistribution(self._distribution)
        qd.setQueryLogLikelihood(True)
        self._query_distribution = qd
        inference_engine.getQueryDistributions().add(qd)

    def results(self, inference_engine, query_output):
        result = {}
        ll = self._query_distribution.getLogLikelihood()
        value = ll.floatValue() if ll is not None else np.nan
        if self._append_variable_names:
            name = ":".join(self._variable_names) + self._column_name
        else:
            name = self._column_name

        result.update({name: value})
        return result

    def reset(self):
        self._query_distribution = None
        self._distribution = None

    def __str__(self):
        return "{}: {}".format(__name__, ", ".join(self._variable_names))


class QueryMeanVariance(QueryBase):
    def __init__(self, variable_name, retract_evidence=True, result_mean_suffix='_mean',
                 result_variance_suffix='_variance', output_dtype=None, default_value=np.nan):

        self._variable_name = variable_name
        self._default_value = default_value
        self._result_mean_suffix = result_mean_suffix
        self._result_variance_suffix = result_variance_suffix
        self._retract_evidence = retract_evidence
        self._output_dtype = output_dtype
        self._variable = None

    def setup(self, network, inference_engine, query_options):
        self._variable = bayesianpy.network.get_variable(network, self._variable_name)

        if not bayesianpy.network.get_variable(network, self._variable_name):
            raise ValueError("Variable {} does not exist in the network".format(self._variable_name))

        if not bayesianpy.network.is_variable_continuous(self._variable):
            raise ValueError("{} needs to be continuous.".format(self._variable_name))

        self._query = bayesServer().CLGaussian(self._variable)

        if self._retract_evidence:
            query_options.setQueryEvidenceMode(bayesServerInference().QueryEvidenceMode.RETRACT_QUERY_EVIDENCE)

        inference_engine.getQueryDistributions().add(bayesServerInference().QueryDistribution(self._query))

    def results(self, inference_engine, query_output):
        mean = self._query.getMean(self._variable)

        if self._output_dtype is not None:
            mean = bayesianpy.data.DataFrame.cast2(self._output_dtype, mean)

        if np.isnan(mean):
            return {self._variable_name + self._result_mean_suffix: self._default_value,
                    self._variable_name + self._result_variance_suffix: self._default_value}

        return {self._variable_name + self._result_mean_suffix: mean,
                self._variable_name + self._result_variance_suffix: self._query.getVariance(self._variable)}

    def reset(self):
        self._query = None
        self._variable = None

    def __str__(self):
        return "P({})".format(self._variable_name)


class QueryKLDivergence(QueryBase):

    def __init__(self, variable_a, variable_b):
        self._variable_a_name = variable_a
        self._variable_b_name = variable_b

    def setup(self, network, inference_engine, query_options):
        distributions = []
        variables = []
        for variable_name in [self._variable_a_name, self._variable_b_name]:
            variable = bayesianpy.network.get_variable(network, variable_name)

            if not bayesianpy.network.get_variable(network, variable_name):
                raise ValueError("Variable {} does not exist in the network".format(variable_name))

            if bayesianpy.network.is_variable_continuous(variable_name):
                distributions.append(bayesServer().CLGaussian(variable))
            else:
                distributions.append(bayesServer().Table(variable))

            variables.append(variable)

        for query in distributions:
            inference_engine.getQueryDistributions().add(bayesServerInference().QueryDistribution(query))

        query_options.setQueryEvidenceMode(bayesServerInference().QueryEvidenceMode.RETRACT_QUERY_EVIDENCE)

        self._distributions = distributions
        self._variables = variables

    def results(self, inference_engine, query_output):

        a = self._distributions[0]
        b = self._distributions[1]

        kl = bayesServerStatistics().KullbackLiebler.Divergence(a, b, bayesServerStatistics().LogarithmBase.Natural)

        return {self._variable_a_name + "_" + self._variable_b_name: kl}

    def reset(self):
        self._distributions = None
        self._variables = None


def _batch_query(df: pd.DataFrame, network_string: str,
                 variable_references: List[str],
                 queries: List[QueryFactory],
                 create_data_reader_command: bayesianpy.reader.CreatableWithDf,
                 create_data_reader_options: bayesianpy.reader.Creatable,
                 logger: logging.Logger = None,
                 ):
    if logger is None:
        logger = logging.getLogger(__name__)

    query_instances = [query.create() for query in queries]

    try:
        bayesianpy.jni.attach(heap_space='1g')
        schema = bayesianpy.data.DataFrame.get_schema(df)

        # TODO: this is very nasty. Need to do this better.
        # DaskDataset (if using Dask) requires a non empty dataframe. Whereas Pandas and DB
        # datasets have this instantiated before being passed in.
        if not df.empty:
            drc = create_data_reader_command.create(df)
        else:
            drc = create_data_reader_command.create(None)

        # TODO: also not great, maybe need a 'callable' on dataReaderCommand?
        if isinstance(drc, jp.JProxy):
            data_reader = drc.getCallable('executeReader')()
        else:
            data_reader = drc.executeReader()

        network = bayesianpy.network.create_network_from_string(network_string)
        reader_options = create_data_reader_options.create()
        variable_refs = list(bayesianpy.network.create_variable_references(network, schema,
                                                                           variable_references=variable_references))
        if len(variable_refs) == 0:
            raise ValueError("Could not match any variables in the supplied dataset with the network. Is it the same?")

        reader = bayesServer().data.DefaultEvidenceReader(data_reader, jp.java.util.Arrays.asList(variable_refs),
                                                          reader_options)

        inference_engine = bayesianpy.model.InferenceEngine(network).create_engine()
        query_options = bayesianpy.model.InferenceEngine.get_inference_factory().createQueryOptions()
        query_output = bayesianpy.model.InferenceEngine.get_inference_factory().createQueryOutput()

        for query in query_instances:
            query.setup(network, inference_engine, query_options)

        ev = bayesianpy.model.Evidence(network, inference_engine).apply()

        results = []

        i = 0
        try:
            while reader.read(ev, bayesServer().data.DefaultReadOptions(True)):
                result = {}

                try:
                    inference_engine.query(query_options, query_output)
                except BaseException as e:
                    logger.error(e)
                    # inference_engine.getEvidence().clear()
                    # continue

                for query in query_instances:
                    result = {**result, **query.results(inference_engine, query_output)}

                ev.clear()
                result.update({'caseid': int(reader.getReadInfo().getCaseId().toString())})

                results.append(result)

                if i % 500 == 0:
                    logger.info("Queried case {}".format(i))

                i += 1
        except BaseException as e:
            logger.error("Unexpected Error!")
            logger.error(e)
        finally:
            reader.close()
            # bayespy.jni.detach()
        if len(results) == 0:
            return pd.DataFrame()

        return pd.DataFrame(results).set_index('caseid')


    except BaseException as e:
        q = [str(query) for query in query_instances]

        logger.error("Unexpected Error: {}. Using queries: {}".format(e, r"\n ".join(q)))


class BatchQuery:
    def __init__(self, network, datastore: bayesianpy.data.DataSet, logger: logging.Logger):
        self._logger = logger
        self._datastore = datastore
        # serialise the network as a string.
        if isinstance(network, bayesianpy.network.Network):
            self._network = network.to_xml()
        else:
            from xml.dom import minidom
            nt = network.saveToString()
            reparsed = minidom.parseString(nt)
            self._network = reparsed.toprettyxml(indent="  ")

    def _calc_num_threads(self, df_size: int, query_size: int, max_threads=None) -> int:
        num_queries = df_size * query_size

        if mp.cpu_count() == 1:
            max = 1
        else:
            max = mp.cpu_count() - 1

        calc = int(num_queries / 5000)
        if calc > max:
            r = max
        elif calc <= 1:
            if num_queries > 1000:
                r = 2
            else:
                r = 1
        else:
            r = calc

        if max_threads is not None and r > max_threads:
            return max_threads

        return r

    def query(self, queries: List[QueryFactory] = None, append_to_df=True,
              variable_references: List[str] = None, max_threads=None):

        if not hasattr(queries, "__getitem__"):
            queries = [queries]

        if variable_references is None:
            variable_references = []

        if queries is None:
            queries = [QueryModelStatistics()]

        nt = self._network
        processes = self._calc_num_threads(len(self._datastore.data), len(queries), max_threads=max_threads)

        self._logger.info("Using {} processes to query {} rows".format(processes, len(self._datastore.data)))

        df = self._datastore.get_dataframe()
        schema = bayesianpy.data.DataFrame.get_schema(df)

        if processes == 1:
            pdf = _batch_query(schema, nt,
                               variable_references, queries,
                               self._datastore.create_data_reader_command(),
                               self._datastore.get_reader_options())
        else:
            # bit nasty, but the only way I could get jpype to stop hanging in Linux.
            ctx._force_start_method('spawn')
            ro = self._datastore.get_reader_options()

            commands = []
            for group in np.array_split(self._datastore.get_dataframe(), processes):
                subset = self._datastore.subset(group.index.tolist())
                commands.append(subset.create_data_reader_command())

            with mp.Pool(processes=processes) as pool:
                pdf = pd.DataFrame()

                # logger with StreamHandler does not pickle, so best to leave it out as an option.
                for result_set in pool.map(lambda drc: _batch_query(schema, nt, variable_references, queries,
                                                                    drc, ro), commands):
                    pdf = pdf.append(result_set)

        if append_to_df:
            return self._datastore.get_dataframe().join(pdf)
        else:
            return pdf


class DaskBatchQuery:
    def __init__(self, network, datastore: bayesianpy.data.DaskDataset):
        self._logger = logging.getLogger(__name__)
        self._datastore = datastore
        # serialise the network as a string.
        if isinstance(network, bayesianpy.network.Network):
            self._network = network.to_xml()
        else:
            from xml.dom import minidom
            nt = network.saveToString()
            reparsed = minidom.parseString(nt)
            self._network = reparsed.toprettyxml(indent="  ")

        if not isinstance(datastore.get_dataframe(), dd.DataFrame):
            raise ValueError("Dataframe has to be of type Dask.DataFrame")

    def _calc_num_threads(self, df_size: int, query_size: int, max_threads=None) -> int:
        num_queries = df_size * query_size

        if mp.cpu_count() == 1:
            max = 1
        else:
            max = mp.cpu_count() - 1

        calc = int(num_queries / 5000)
        if calc > max:
            r = max
        elif calc <= 1:
            if num_queries > 1000:
                r = 2
            else:
                r = 1
        else:
            r = calc

        if max_threads is not None and r > max_threads:
            return max_threads

        return r

    def _generate_dask_metadata(self, row, variable_references, queries) -> pd.DataFrame:
        meta = _batch_query(row, self._network, variable_references, queries,
                            self._datastore.create_data_reader_command(),
                            self._datastore.get_reader_options())

        # row = row.append(meta)

        # for query in queries:
        # empty JPype references out, so we don't try and pickle them later.
        #    query.reset()

        return bayesianpy.data.DataFrame.get_schema(meta)

    def query(self, queries: List[QueryFactory] = None, append_to_df=True,
              variable_references: List[str] = None, max_threads=None):

        if not hasattr(queries, "__getitem__"):
            queries = [queries]

        if variable_references is None:
            variable_references = []

        if queries is None:
            queries = [QueryModelStatistics()]

        nt = self._network
        dk = self._datastore.get_dataframe()

        metadata = self._generate_dask_metadata(dk.head(1), variable_references, queries)

        drc = self._datastore.create_data_reader_command()
        ro = self._datastore.get_reader_options()

        results = dk.map_partitions(_batch_query, network_string=nt,
                                    variable_references=variable_references,
                                    queries=queries,
                                    create_data_reader_command=drc,
                                    create_data_reader_options=ro,
                                    logger=self._logger,
                                    meta=metadata)
        # .compute(get=multiprocessing.get,
        #                                               num_workers=self._calc_num_threads(len(dk), len(queries),
        #                                               max_threads=max_threads))
        if append_to_df:
            return dk.join(results)
        else:
            return results
