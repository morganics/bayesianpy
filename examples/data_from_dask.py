import bayesianpy

import logging
import os

from sklearn.metrics import accuracy_score
import dask.dataframe as dd
import bayesianpy.dask as dk

def main():

    logger = logging.getLogger()

    bayesianpy.jni.attach(logger)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    bayesianpy.jni.attach(logger)

    db_folder = bayesianpy.utils.get_path_to_parent_dir(__file__)
    titanic_dask = dd.read_csv(os.path.join(db_folder, "data/titanic.csv"))

    auto = bayesianpy.data.AutoType(titanic_dask)
    network_factory = bayesianpy.network.NetworkFactory(logger)

    discrete = auto.get_discrete_variables()
    continuous = auto.get_continuous_variables()

    # write data to the temporary sqllite db
    with bayesianpy.data.DataSet(titanic_dask, db_folder, logger) as dataset:
        # learn the model structure using built-in algorithm

        # Or, use a standard template, which generally gives good performance
        mixture_naive_bayes_tpl = bayesianpy.template.MixtureNaiveBayes(logger, discrete=titanic_dask[discrete],
                                                                        continuous=titanic_dask[continuous])

        model = bayesianpy.model.NetworkModel(
                mixture_naive_bayes_tpl.create(network_factory),
                logger)

        # result contains a bunch of metrics regarding the training step
        model.train(dataset.subset(dk.compute(titanic_dask.index).tolist()))

        # note that we've not 'dropped' the target data anywhere, this will be retracted when it's queried,
        # by specifying query_options.setQueryEvidenceMode(bayesServerInference().QueryEvidenceMode.RETRACT_QUERY_EVIDENCE)
        results = model.batch_query(dataset.subset(dk.compute(titanic_dask.index).tolist()), [
                bayesianpy.model.QueryMostLikelyState("Survived", output_dtype=titanic_dask['Survived'].dtype)])

        # Each query just appends a column/ columns on to the original dataframe, so results is the same as titanic.iloc[test_indexes],
        # with (in this case) one additional column called 'Survived_maxlikelihood', joined to the original.
        score = accuracy_score(y_pred=dk.compute(results['Survived_maxlikelihood']).tolist(),
                                    y_true=dk.compute(results['Survived']).tolist())

        logger.info("Score was {}.".format(score))


if __name__ == "__main__":
    main()