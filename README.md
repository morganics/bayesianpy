# BayesPy

A Python SDK for (primarily) feature selection using the BayesServer Java API

# Example Usage on Titanic

``` python
logger = logging.getLogger('variable_selection_wrapper')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# get a good guess as to whether the variable is discrete/ continuous
auto = bayespy.data.AutoType(titanic)

# create the network factory that can be used to instantiate new networks
with bayespy.network.NetworkFactory(titanic, logger) as network_factory:
    # create the insight model
    insight = bayespy.insight.AutoInsight(network_factory, logger, discrete=titanic[list(auto.get_discrete_variables())],         continuous=titanic[list(auto.get_continuous_variables())])

    # iterate over the best combination of variables by assessing lots of combinations of comparison queries
    results = insight.query_variable_combinations(bayespy.network.Discrete("Survived", 0))

    # results here is a list of dicts, containing the keys: probability (the percentage of cases that the model accounts for), model (the trained model), evidence (the names of the variable+states).

    sorted_combos = sorted(top_variable_combinations, key=lambda x: x['probability'], reverse=True)
    top_combos = [(sc['evidence'], sc['probability']) for sc in sorted_combos if sc['probability'] > 0.89]
```

# Namespaces

1. Data - Python data related tasks, AutoType, DataFrame utility methods etc
2. JNI - The Java/ JPype Python bridge setup
3. ML - Not used much at the moment, but 'wrapper' approaches for iterative variable selection
4. Model - the trained model used for querying
5. Network - network creation / structural classes
6. Visual - Classes for visualising the queries
