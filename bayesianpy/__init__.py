from bayesianpy import data
from bayesianpy import insight
#from bayespy import ml
from bayesianpy import model
from bayesianpy import network
from bayesianpy import template
from bayesianpy import visual
from bayesianpy.jni import bayesServer as _bs
from bayesianpy import utils

def license(key):
    l = _bs().License.validate(key)