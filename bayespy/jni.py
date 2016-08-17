import jpype as jp
import os
import bayespy.utils
import platform

path_to_package = bayespy.utils.get_path_to_parent_dir(__file__)
separator = ";"
if platform.system() == "Linux":
    separator = ":"

classpath = ".{0}{1}{0}{2}".format(separator, os.path.join(path_to_package, 'bin/bayesserver-7.6.jar'),
                              os.path.join(path_to_package, 'bin/sqlite-jdbc-3.8.11.2.jar'))
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), "-Djava.class.path={}".format(classpath))
    bayesServer = jp.JPackage("com.bayesserver")
    sqlLite = jp.JPackage("org.sqlite.JDBC")
    bayesServerInference = jp.JPackage("com.bayesserver.inference")
    bayesServerParams = jp.JPackage("com.bayesserver.learning.parameters")