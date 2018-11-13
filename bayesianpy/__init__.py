
#from bayesianpy import insight

from bayesianpy import distributed
#from bayespy import ml
from bayesianpy import model
from bayesianpy import reader
from bayesianpy import output
from bayesianpy import network
from bayesianpy import template
from bayesianpy import data
from bayesianpy import jni
#from bayesianpy import visual
from bayesianpy import utils
from bayesianpy.jni import bayesServer as _bs

#from bayesianpy import analysis
#from bayesianpy import data

def license(key):
    l = _bs().License.validate(key)