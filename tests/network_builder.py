import logging
import unittest.mock as mock
import unittest
import bayesianpy.network
import tests.iris

def create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger

class NetworkBuilderTestCase(unittest.TestCase):

    def setUp(self):
        bayesianpy.jni.attach()
        # Setup fake redis for testing.
        pass

    def tearDown(self):
        # Clear data in fakeredis.
        pass

    def test_build_simple_network(self):
        df = tests.iris.create_iris_dataset()

        network = bayesianpy.network.Network.from_new()
        network, _ = network.nodes().add(['sepal_length']).continuous(using=df)

        self.assertEquals(len(network.nodes()), 5)


if __name__ == "__main__":
    unittest.main()
