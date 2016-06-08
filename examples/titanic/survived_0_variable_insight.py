import pandas as pd
import numpy as np
import re
import bayespy
import logging
import sys

titanic = pd.read_csv("data/train.csv")
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

if len(sys.argv) > 1:
    bayespy.license(sys.argv[1])
else:
    print("Will not be able to save network files, as this is a trial version.")

logger = logging.getLogger('variable_selection_wrapper')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# get a good guess as to whether the variable is discrete/ continuous
auto = bayespy.data.AutoType(titanic)

# create the network factory that can be used to instantiate networks
network_factory = bayespy.network.NetworkFactory(titanic, logger)

# create the insight model
insight = bayespy.insight.AutoInsight(network_factory, discrete=titanic[list(auto.get_discrete_variables())], continuous=titanic[list(auto.get_continuous_variables())])

# iterate over the best combination of variables by assessing lots of combinations of comparison queries
results = insight.query(bayespy.network.Discrete("Survived", "0"))

base_evidence_false = insight.rationalise(results, num=6)

# get unique set of best variables
bef_set = set(el[0] for el in base_evidence_false)

import numpy as np
import matplotlib.pyplot as plt

import math
import matplotlib.mlab as mlab
#import uuid
import seaborn as sns

# plot the most significant results, where continuous results will have the most significant states highlighted.
for r in results:
    model = r['model']
    for item in r['evidence']:
        if item not in bef_set:
            continue

        if bayespy.network.is_cluster_variable(item):
            fig1 = plt.figure()
            ax = fig1.add_subplot(211)
            node = bayespy.network.Discrete.fromstring(item)
            v_name = node.variable.replace("Cluster_","")
            min_x = titanic[v_name].min()
            max_x = titanic[v_name].max()
            step = int((max_x - min_x) / 150)

            if step == 0:
                step = 2

            for i in range(3):
                color = "b"
                if "Cluster{}".format(i) in item:
                    color = "r"

                state = "Cluster{}".format(i)
                e = bayespy.network.Discrete(node.variable, state)
                #(d, c) = insight.evidence_query(model=model, base_evidence=[bayespy.state("vds", "1")], new_evidence=[bayespy.state("vds", "0")])
                (d, c) = insight.evidence_query(model=model, new_evidence=[e.tostring()])
                mu = c[c.variable == e.variable.replace("Cluster_", "")].iloc[0]['mean']
                sigma = math.sqrt(c[c.variable == e.variable.replace("Cluster_", "")].iloc[0]['variance'])
                x = np.linspace(min_x, max_x, num=150)
                ax.plot(x,mlab.normpdf(x, mu, sigma), color=color)
                ax.set_xlim([min_x - step, max_x + step])
            ax.set_title(v_name)
            ax = fig1.add_subplot(212)

            sns.distplot(titanic[titanic.Survived == 0][v_name].dropna(), ax=ax, bins=range(int(min_x), int(max_x), step))
            sns.distplot(titanic[titanic.Survived == 1][v_name].dropna(), ax=ax, bins=range(int(min_x), int(max_x), step))

            ax.set_xlim([min_x - step, max_x + step])
            #plt.savefig("..\\plots\\auto\\{}_{}.png".format(v_name, uuid.uuid4()))
        else:
            n = bayespy.network.Discrete.fromstring(item)
            plt.figure()
            sns.countplot(x=n.variable, hue='Survived', data=titanic)
            #plt.savefig("..\\plots\\auto\\{}.{}_{}.png".format(n.variable, n.state, uuid.uuid4()))