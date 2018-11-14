import jpype as jp
import os
import bayesianpy.directory_utils
import platform

BAYES_SERVER_VERSION = "7.24"

def attach_thread(logger=None):
    if not jp.isThreadAttachedToJVM():
        if logger is not None:
            logger.debug("Attaching thread to JVM")
        jp.attachThreadToJVM()

def attach(logger=None, heap_space='6g'):
    if logger is not None:
        logger.debug("JVM Started: {}".format(jp.isJVMStarted()))

    if not jp.isJVMStarted():
        path_to_package = bayesianpy.directory_utils.get_path_to_parent(__file__)
        separator = ";"
        if platform.system() == "Linux":
            separator = ":"

        jars = ['bin/bayesserver-{}.jar'.format(BAYES_SERVER_VERSION), 'bin/sqlite-jdbc-3.8.11.2.jar',
                'bin/mysql-connector-java-5.0.8-bin.jar', 'bin/jaybird-full-3.0.2.jar']
        classpath = ".{}".format(separator)
        for jar in jars:
            if os.path.exists(os.path.join(path_to_package, jar)):
                classpath += "{}{}".format(os.path.join(path_to_package, jar), separator)

        if logger is not None:
             logger.debug("Starting JVM ({})...".format(classpath))

        jp.startJVM(jp.getDefaultJVMPath(), "-Djava.class.path={}".format(classpath), "-XX:-UseGCOverheadLimit", "-Xmx{}".format(heap_space))

        if logger is not None:
             logger.debug("JVM Started.")
        # so it doesn't crash if called by a Python thread.
    attach_thread(logger)

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

def bayesServerSampling():
    return jp.JPackage("com.bayesserver.data.sampling")

def bayesServerStatistics():
    return jp.JPackage("com.bayesserver.statistics")
