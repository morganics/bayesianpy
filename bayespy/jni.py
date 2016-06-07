import jpype as jp
import os

base = os.path.dirname(os.path.realpath('__file__'))
_classpath = ".;{};{}".format(os.path.join(base, 'bayespy/bin/bayesserver-7.6.jar'),
                              os.path.join(base, 'bayespy/bin/sqlite-jdbc-3.8.11.2.jar'))
print(_classpath)
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), "-Djava.class.path=%s" % _classpath)
    bayesServer = jp.JPackage("com.bayesserver")
    sqlLite = jp.JPackage("org.sqlite.JDBC")
    bayesServerInference = bayesServer.inference
    bayesServerParams = jp.JPackage("com.bayesserver.learning.parameters")