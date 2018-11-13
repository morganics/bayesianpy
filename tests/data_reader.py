import logging
import unittest.mock as mock
import unittest
import bayesianpy.network
import pandas as pd
import bayesianpy.utils.list
import tests.iris
import dask.dataframe as dd
import bayesianpy.reader
from timeit import default_timer as timer

class TestDaskDataReader(unittest.TestCase):

    def test_pre_loading(self):
        npartitions = 10
        all_ddf = dd.concat([dd.from_pandas(tests.iris.create_iris_dataset(), npartitions=npartitions) for i in range(0, 500)], interleave_partitions=True)
        #print(all_ddf)
        total_length = len(all_ddf)
        length_of_partition = len(all_ddf.get_partition(0).compute())
        bayesianpy.jni.attach()
        dfr = bayesianpy.reader.PandasDataReaderCommand(all_ddf, preload=False)
        reader = dfr.executeReader()
        start = timer()
        for i in range(0, total_length):
            reader.getCallable('read')()
        end = timer()

        non_preloaded = end-start

        dfr = bayesianpy.reader.PandasDataReaderCommand(all_ddf, preload=True)
        reader = dfr.executeReader()
        start = timer()
        for i in range(0, total_length):
            reader.getCallable('read')()
        end = timer()

        preloaded = end-start
        print(non_preloaded)
        print(preloaded)
        self.assertTrue(preloaded < non_preloaded)
        #self.assertEqual(0, reader.getCallable("get_partition_index")())
        self.assertEqual(1, reader.getCallable("get_loaded_partition_index")())

        for i in range(0, total_length - (length_of_partition+10)):
            reader.getCallable('read')()

        self.assertEqual(npartitions, reader.getCallable("get_partition_index")())
        self.assertEqual(npartitions, reader.getCallable("get_loaded_partition_index")())
        #

        #     current_partition = reader.getCallable('get_partition')()
        #     print(current_partition)
        #     row_value = reader.getCallable('getString')(0)
        #     print(row_value)