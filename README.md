# BayesPy

A Python SDK for performing common operations on the Bayes Server Java API, and trying to utilise the best of Python (e.g. dataframes and visualisation). This wraps calls to the Java API with Jpype1.

Supported functionality (currently only supports contemporal networks, although Bayes Server supports temporal networks as well):

 - Creating network structures (in network.py and template.py, discrete/ continuous/ discretised/ multivariate nodes)
 - Training a model (in model.py)
 - Querying a model with common query types such as LogLikelihood/ conflict queries, joint probabilities for both continuous and discrete variables (in model.py, allows for multiprocessing as well to speed up query times)
 - AutoInsight (using difference queries to understand variables' significance to the model, in insight.py)
 - Various utility functions for reading dataframes, casting and generally mapping between dataframes -> SQLlite -> Bayes Server.
 
Note: I believe there is now an in-memory implementation for mapping between dataframes and Bayes Server, however the SDK currently writes data to an SQLlite database which is then read by the Java API.

# Example; training a model from a template:

``` python

logger = logging.getLogger()

# utility function to decide on whether variables are discrete/ continuous
# df is a pandas dataframe.
auto = bayespy.data.AutoType(df)

# creates a template to create a single discrete cluster (latent) node with edges to independent 
# child nodes
tpl = bayespy.template.MixtureNaiveBayes(logger,
                                                 discrete=df[list(auto.get_discrete_variables())],
                                                 continuous=df[list(auto.get_continuous_variables())],
                                                 latent_states=8)

with bayespy.network.NetworkFactory(df, self._db_folder, self._logger) as nf:
    model = bayespy.model.NetworkModel(tpl.create(nf), nf.get_datastore(), logger)
    model.train()
    model.save("model.bayes")
```

# Example; querying a model:
``` python

with bayespy.network.NetworkFactory(df, self._db_folder, logger, network_file_path='model.bayes') as nf:
    model = bayespy.model.NetworkModel(nf.create(), nf.get_datastore(), logger)
    
    # Get the loglikelihood of the model given the evidence specified in df (here, using the same data as was trained upon)
    # Can also specify to calculate conflict, if required.
    # 'results' is a pandas dataframe, where each variable in df will have an additional column with a suffix of _loglikelihood.
    results = model.batch_query(bayespy.model.QueryModelStatistics())
        
```    
   

