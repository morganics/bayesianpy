# BayesPy

A Python SDK for performing common operations on the Bayes Server Java API, and trying to utilise the best of Python (e.g. dataframes and visualisation). This wraps calls to the Java API with Jpype1. This wrapper is not released/ supported/ endorsed by Bayes Server.

Supported functionality (currently only supports contemporal networks, although Bayes Server supports temporal networks as well):

 - Creating network structures (in network.py and template.py, discrete/ continuous/ discretised/ multivariate nodes)
 - Training a model (in model.py)
 - Querying a model with common query types such as LogLikelihood/ conflict queries, joint probabilities for both continuous and discrete variables (in model.py, allows for multiprocessing as well to speed up query times)
 - AutoInsight (using difference queries to understand variables' significance to the model, in insight.py)
 - Various utility functions for reading dataframes, casting and generally mapping between dataframes -> SQLlite -> Bayes Server.
 
Note: I believe there is now an in-memory implementation for mapping between dataframes and Bayes Server, however the SDK currently writes data to an SQLlite database which is then read by the Java API.

## Motivation

Python is a simpler language to put together something quickly, the Bayes Server API is very powerful, and consequently can be time consuming to work with directly. I haven't tried to wrap every single piece of Java code, however I have tried to - in general - separate out any Java calls from the client of the SDK, to allow type hinting and remove any confusion of working through Jpype. You can do a lot more with the Java API directly, however the most common usage; creating network structures, training and querying networks should be mostly accounted for. The Java API is fairly stable (e.g. it doesn't change very much from release to release) however this Python wrapper is very much in flux!

## Jupyter examples

- [Titanic Classification example] (http://htmlpreview.github.io/?https://github.com/morganics/BayesPy/blob/master/examples/notebook/titanic_classification.slides.html) provides a brief walkthrough of how to construct a network and run a batch query while using cross validation
- [Iris Anomaly detection example] (http://htmlpreview.github.io/?https://github.com/morganics/BayesPy/blob/master/examples/notebook/iris_anomaly_detection.slides.html) provides a brief walkthrough  training a manually crafted network, as well as a batch query to obtain the Log Likelihood information theoretic score from the trained model to assist in identifying 'abnormal' data.
- [Iris cluster visualisation with covariance] (http://htmlpreview.github.io/?https://github.com/morganics/BayesPy/blob/master/examples/notebook/iris_gaussian_mixture_model.slides.html) provides a brief walkthrough  training a naive Bayes network followed by a fully connected Gaussian mixture model, and how the clustering/ classification is affected as a result.

## Example: training a model from a template

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

network_factory = bayespy.network.NetworkFactory(logger)
with bayespy.data.DataSet(df, db_folder, logger) as dataset:
    model = bayespy.model.NetworkModel(tpl.create(network_factory), dataset, logger)
    model.train()
    model.save("model.bayes")
```

## Example: querying a model
``` python

# specify the filename of the trained model
network_factory = bayespy.network.NetworkFactory(logger, network_file_path='model.bayes')
with bayespy.data.DataSet(df, db_folder, logger) as dataset:
    model = bayespy.model.NetworkModel(network_factory.create(), dataset, logger)    
    # Get the loglikelihood of the model given the evidence specified in df (here, using the same data as was trained upon)
    # Can also specify to calculate conflict, if required.
    # 'results' is a pandas dataframe, where each variable in df will have an additional column with a suffix of _loglikelihood.
    results = model.batch_query(bayespy.model.QueryModelStatistics())
        
```    
## More examples

A classification and regression example are included in the examples folder on the Titanic dataset. I'll try and put some more up shortly. 

