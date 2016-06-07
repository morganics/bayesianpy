import pandas as pd
titanic = pd.read_csv("titanic/data/train.csv")
import numpy as np
import re

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
titanic.drop(['Cabin', 'Ticket', 'Name'], inplace=True, axis=1)

import bayespy
import logging
import sys

bayespy.license(sys.argv[1])

logger = logging.getLogger('variable_selection_wrapper')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

auto = bayespy.data.AutoType(titanic)

network_factory = bayespy.network.NetworkFactory(titanic, logger)
insight = bayespy.insight.AutoInsight(network_factory, discrete=titanic[list(auto.get_discrete_variables())], continuous=titanic[list(auto.get_continuous_variables())])

results = insight.query(bayespy.network.DiscreteNode("Survived", "0"))

base_evidence_false = insight.rationalise(results, num=6)
bef_set = set(el[0] for el in base_evidence_false)