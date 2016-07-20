from bayespy import data
from bayespy import insight
#from bayespy import ml
from bayespy import model
from bayespy import network
#from bayespy import visual
from bayespy.jni import bayesServer as _bs
from bayespy import utils

def license(key):
    l = _bs.License.validate(key)