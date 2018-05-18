import logging
import unittest.mock as mock
import unittest
import bayesianpy.network
import pandas as pd
import bayesianpy.utils.list

class TestList(unittest.TestCase):

    def test_except(self):
        lst = [0, 1, 2, 3, 4]
        parts = bayesianpy.utils.list.exclude(lst, 3)
        self.assertListEqual([0, 1, 2, 4], parts)

    def test_except_with_0_index(self):
        lst = [0, 1, 2, 3, 4]
        parts = bayesianpy.utils.list.exclude(lst, 0)
        self.assertListEqual([1, 2, 3, 4], parts)

    def test_except_with_end_index(self):
        lst = [0, 1, 2, 3, 4]
        parts = bayesianpy.utils.list.exclude(lst, 4)
        self.assertListEqual([0, 1, 2, 3], parts)