import jpype as jp
import os

path_to_package = os.path.dirname(os.path.abspath(__file__))

classpath = ".;{};{}".format(os.path.join(path_to_package, 'bin/bayesserver-7.6.jar'),
                              os.path.join(path_to_package, 'bin/sqlite-jdbc-3.8.11.2.jar'))
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), "-Djava.class.path=%s" % classpath)
    bayesServer = jp.JPackage("com.bayesserver")
    sqlLite = jp.JPackage("org.sqlite.JDBC")
    bayesServerInference = bayesServer.inference
    bayesServerParams = jp.JPackage("com.bayesserver.learning.parameters")