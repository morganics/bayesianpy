import pandas as pd
import numpy as np
import re
import bayespy
import logging
import sys
import os

db_folder = bayespy.utils.get_path_to_parent_dir(__file__)
titanic = pd.read_csv(os.path.join(db_folder, "data/train.csv"))
pattern = re.compile("([A-Z]{1})([0-9]{1,3})")


def get_cabin_floor_and_number(cabin):
    if not isinstance(cabin, str):
        return "", np.nan

    cabins = cabin.split(" ")
    for cabin in cabins:
        match = re.match(pattern, cabin)
        if match is not None:
            floor = match.group(1)
            number = match.group(2)

            return floor, number
    return "", np.nan

titanic['Floor'], titanic['CabinNumber'] = zip(*titanic.Cabin.map(get_cabin_floor_and_number))
titanic.CabinNumber = titanic.CabinNumber.astype(float)
titanic.Floor.replace("", np.nan, inplace=True)
titanic.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True, axis=1)

if len(sys.argv) > 1:
    bayespy.license(sys.argv[1])
else:
    print("Will not be able to save network files, as this is a trial version.")

logger = logging.getLogger('variable_selection_wrapper')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# get a good guess as to whether the variable is discrete/ continuous
auto = bayespy.data.AutoType(titanic)
target = bayespy.network.Discrete("Survived", 0)


# create the network factory that can be used to instantiate networks
with bayespy.network.NetworkFactory(titanic, logger, db_folder) as network_factory:
    # create the insight model
    insight = bayespy.insight.AutoInsight(network_factory, logger, discrete=titanic[list(auto.get_discrete_variables())], continuous=titanic[list(auto.get_continuous_variables())])
    models = insight.create_model_cache(target, times=5)
    print(insight.query_bivariate_combinations(target, models=models, top=10))
    print(insight.query_exclusive_states(target, models=models, top=10))

    combinations = list(insight.query_top_variable_combinations_as_df(target, models=models, top=4))
    print(combinations)
