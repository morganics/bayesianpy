import jpype as jp
import os
import bayespy.utils
import platform

def attach_thread():
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

def attach():
    if not jp.isJVMStarted():
        path_to_package = bayespy.utils.get_path_to_parent_dir(__file__)
        separator = ";"
        if platform.system() == "Linux":
            separator = ":"

        classpath = ".{0}{1}{0}{2}".format(separator, os.path.join(path_to_package, 'bin/bayesserver-7.8.jar'),
                                           os.path.join(path_to_package, 'bin/sqlite-jdbc-3.8.11.2.jar'))

        jp.startJVM(jp.getDefaultJVMPath(), "-Djava.class.path={}".format(classpath), "-XX:-UseGCOverheadLimit", "-Xmx6g")

        # so it doesn't crash if called by a Python thread.
        attach_thread()

def detach():
    if jp.isThreadAttachedToJVM():
        jp.detachThreadFromJVM()

def bayesServer():
    return jp.JPackage("com.bayesserver")

def bayesServerInference():
    return jp.JPackage("com.bayesserver.inference")

def bayesServerAnalysis():
    return jp.JPackage("com.bayesserver.analysis")

def bayesServerParams():
    return jp.JPackage("com.bayesserver.learning.parameters")

def bayesServerDiscovery():
    return jp.JPackage("com.bayesserver.data.discovery")

def bayesServerStructure():
    return jp.JPackage("com.bayesserver.learning.structure")

