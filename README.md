# BayesianPy

## Disclaimer
**This wrapper is not released/ supported/ endorsed by Bayes Server.** Please look at the examples on https://www.bayesserver.com/code/python/setup-py to understand how to implement the integration. These samples here might be out of date and do not keep up with the release cycle of the Bayes Server package (and also don't follow good software engineering practices). They might be useful as a reference, but will not (always/ typically) work out of the box. I am now doing some work on wrapping the .NET core DLLs which I've found to be a simpler process.
 
## Information
A Python SDK for performing common operations on the Bayes Server Java API, and trying to utilise the best of Python (e.g. dataframes and visualisation). This wraps calls to the Java API with Jpype1 (this library has now been updated and some of the examples do things that you no longer need to worry about).

Supported functionality (currently only supports contemporal networks, although Bayes Server supports temporal networks as well):

 - Creating network structures (in network.py and template.py, discrete/ continuous/ discretised/ multivariate nodes)
 - Training a model (in model.py)
 - Querying a model with common query types such as LogLikelihood/ conflict queries, joint probabilities for both continuous and discrete variables (in model.py, allows for multiprocessing as well to speed up query times)
 - AutoInsight (using difference queries to understand variables' significance to the model, in insight.py)
 - Various utility functions for reading dataframes, casting and generally mapping between dataframes -> SQLlite -> Bayes Server.
 
Note: The SDK currently writes data to an SQLlite database which is then read by the Java API.

## Motivation

Python is a simpler language to put something together quickly and the Bayes Server API is very powerful, but gives you lots of options. I haven't tried to wrap every single piece of Java code, however I have tried to - in general - separate out any Java calls from the client of the SDK, to allow type hinting and remove any confusion of working through Jpype. You can do a lot more with the Java API directly, however the most common usage; creating network structures, training and querying networks should be mostly accounted for. The Java API is stable (e.g. it doesn't change very much from release to release) however this Python wrapper is very much in flux!

## Are Bayesian networks Bayesian? (from BayesServer.com)

Yes and no. They do make use of Bayes Theorem during inference, and typically use priors during batch parameter learning. However they do not typically use a full Bayesian treatment in the Bayesian statistical sense (i.e. hyper parameters and learning case by case).
The matter is further confused, as Bayesian networks tyically DO use a full Bayesian approach for Online learning.

## Jupyter examples

- [Titanic Classification example] (https://github.com/morganics/bayesianpy/blob/master/examples/notebook/titanic_classification.ipynb) provides a brief walkthrough of how to construct a network and run a batch query while using cross validation
- [Iris Anomaly detection example] (https://github.com/morganics/bayesianpy/blob/master/examples/notebook/iris_anomaly_detection.ipynb) provides a brief walkthrough  training a manually crafted network, as well as a batch query to obtain the Log Likelihood information theoretic score from the trained model to assist in identifying 'abnormal' data.
- [Iris cluster visualisation with covariance] (https://github.com/morganics/bayesianpy/blob/master/examples/notebook/iris_gaussian_mixture_model.ipynb) provides a brief walkthrough  training a naive Bayes network followed by a fully connected Gaussian mixture model, and how the clustering/ classification is affected as a result.
- [Iris joint probability PDF visualisation] (https://github.com/morganics/bayesianpy/blob/master/examples/notebook/iris_univariate_joint_pdf_plot.ipynb) Creates a fully connected Gaussian mixture model, where each variable is independently queried given the iris_class. Provides code for plotting a 1D joint distribution.
- [Diabetes Linear regression example] (https://github.com/morganics/bayesianpy/blob/master/examples/notebook/diabetes_linear_regression.ipynb) Creates a simple naive Bayes network to give a linear regression model for the diabetes dataset from scikit-learn, with mean and variance.
- [Diabetes Non-Linear regression example] (https://github.com/morganics/bayesianpy/blob/master/examples/notebook/diabetes_non_linear_regression.ipynb) Creates a mixture of Gaussians network to give a non-linear regression model for the diabetes dataset from scikit-learn, with mean and variance.
- [Iris Cluster Count](https://github.com/morganics/bayesianpy/blob/master/examples/notebook/iris_cluster_count.ipynb) Simple demo to show selecting the optimal number of clusters for a latent variable using the Iris dataset.

## Example: training a model from a template

``` python

logger = logging.getLogger()

# utility function to decide on whether variables are discrete/ continuous
# df is a pandas dataframe.
auto = bayesianpy.data.AutoType(df)

# creates a template to create a single discrete cluster (latent) node with edges to independent 
# child nodes
tpl = bayesianpy.template.MixtureNaiveBayes(logger,
                                                 discrete=df[list(auto.get_discrete_variables())],
                                                 continuous=df[list(auto.get_continuous_variables())],
                                                 latent_states=8)

network_factory = bayesianpy.network.NetworkFactory(logger)
with bayesianpy.data.DataSet(df, db_folder, logger) as dataset:
    model = bayesianpy.model.NetworkModel(tpl.create(network_factory), logger)
    model.train(dataset) # or you can use a subset of the data, e.g. dataset.subset(list_of_indices)
    model.save("model.bayes")
```

## Example: querying a model
``` python

# specify the filename of the trained model
network_factory = bayesianpy.network.NetworkFactory(logger, network_file_path='model.bayes')
with bayesianpy.data.DataSet(df, db_folder, logger) as dataset:
    model = bayesianpy.model.NetworkModel(network_factory.create(), dataset, logger)    
    # Get the loglikelihood of the model given the evidence specified in df (here, using the same data as was trained upon)
    # Can also specify to calculate conflict, if required.
    # 'results' is a pandas dataframe, where each variable in df will have an additional column with a suffix of _loglikelihood.
    results = model.batch_query(dataset, [bayesianpy.model.QueryModelStatistics()])
        
```    
## More examples

A classification and regression example are included in the examples folder on the Titanic dataset. I'll try and put some more up shortly. 

