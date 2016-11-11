import numpy as np
from bayespy.jni import *
import logging
import pandas as pd

class DataFrameReader:
    def __init__(self, df):
        self._df = df
        self._row = None
        self._columns = df.columns.tolist()
        self.reset()
        self.row_index = 0

    def __getattr__(self, key):
        return self.__getitem__(key)

    def read(self):
        self._row = next(self._iterator, None)
        self.row_index += 1
        return self._row is not None

    def reset(self):
        self._iterator = self._df.itertuples()

    def get_index(self):
        return self._row[0]

    def to_dict(self):
        return self.row()

    def tolist(self, cols):
        return [self.__getitem__(c) for c in cols]

    def row(self):
        return {c : self.__getitem__(c) for c in self._columns}

    def __getitem__(self, key):
        # the df index of the row is at index 0
        try:
            ix = self._columns.index(key) + 1
            return self._row[ix]
        except BaseException as e:
            return None

    def __next__(self):
        if self.read():
            return self
        else:
            raise StopIteration

class AutoType:
    def __init__(self, df, discrete=[], continuous=[], continuous_to_discrete_limit = 20):
        self._df = df
        self._continuous_to_discrete_limit = continuous_to_discrete_limit
        self._continuous = continuous
        self._discrete = discrete

    def get_continuous_variables(self):
        cols = self._df.columns.tolist()
        for col in cols:
            if not DataFrame.is_float(self._df[col].dtype) and not DataFrame.is_int(self._df[col].dtype):
                continue
            if col in self._discrete:
                continue
            elif col in self._continuous:
                yield col
            elif len(self._df[col].unique()) > self._continuous_to_discrete_limit:
                yield col

    def get_discrete_variables(self):
        continuous = set(self.get_continuous_variables())
        for col in self._df.columns.tolist():
            if col in self._continuous:
                continue

            if DataFrame.is_timestamp(self._df[col].dtype):
                continue

            elif col in self._discrete:
                yield col
            elif col not in continuous:
                yield col

class DataFrame:
    @staticmethod
    def is_timestamp(dtype):
        return str(dtype) in ['timestamp64', 'timedelta64']

    @staticmethod
    def is_numeric(dtype):
        return DataFrame.is_float(dtype) or DataFrame.is_int(dtype)

    @staticmethod
    def is_float(dtype):
        return str(dtype) in {"float32", "float64"}

    @staticmethod
    def is_int(dtype):
        return str(dtype) in {"int32", "int64"}

    @staticmethod
    def is_bool(dtype):
        return str(dtype) == "bool"

    @staticmethod
    def is_string(dtype):
        return str(dtype) == "object" or str(dtype) == "O"

    @staticmethod
    def could_be_int(col):
        if DataFrame.is_int(col.dtype):
            return True

        if DataFrame.is_float(col.dtype):
            for val in col.dropna().unique():
                if int(val) != val:
                    return False

            return True

        return False

    @staticmethod
    def coerce_to_numeric(df: pd.DataFrame, logger: logging.Logger, cutoff=0.10, ignore=[]):
        for col in df.columns:
            if col in ignore:
                continue
            values = df[col].dropna().unique()
            ratio = 0
            pre_length = len(values)

            if pre_length > 0:
                new_values = pd.to_numeric(df[col].dropna().unique(), errors='coerce')
                post_length = len(new_values[~np.isnan(new_values)])
                ratio = (pre_length - post_length) / pre_length

            if ratio <= cutoff:
                logger.debug("Converting column {} to numeric (ratio: {})".format(col, ratio))
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.debug("Not converting column {} (ratio: {})".format(col, ratio))

        return df

    @staticmethod
    def cast2(dtype, value):
        if DataFrame.is_int(dtype):
            return int(value)
        if DataFrame.is_bool(dtype):
            return value == True
        if DataFrame.is_float(dtype):
            return float(value)

        return value

    @staticmethod
    def cast(df, col_name, value):
        if DataFrame.is_int(df[col_name].dtype):
            return int(value)
        if DataFrame.is_bool(df[col_name].dtype):
            return value == True
        if DataFrame.is_float(df[col_name].dtype):
            return float(value)

        return value

    @staticmethod
    def replace_0_with_normal_dist(df: pd.DataFrame, columns):
        for col in columns:
            df.loc[df[col] == 0, col] = df[col].apply(lambda x: np.random.normal(0, 3))

class Filter:
    @staticmethod
    def remove_static_variables(df: pd.DataFrame):
        column_names = df.apply(lambda x: len(x.unique()) > 1)
        return df[column_names[column_names == True].index.tolist()].copy()

    @staticmethod
    def remove_variable_variables(df: pd.DataFrame):
        column_names = df.apply(lambda x: len(x.unique()) != len(df))
        return df[column_names[column_names == True].index.tolist()]

    @staticmethod
    def remove_discrete_variables_with_too_many_states(df: pd.DataFrame, num_states = 30):
        column_names = df.select_dtypes(include=['object']).apply(lambda x: len(x.unique()) >= num_states)
        cols = list(set(df.columns.tolist()) - set(column_names[column_names == True].index.tolist()))
        return df[cols]

    @staticmethod
    def apply(df: pd.DataFrame):
        return Filter.remove_discrete_variables_with_too_many_states(Filter.remove_variable_variables(Filter.remove_static_variables(df)))

def create_histogram(series):
    hdo = bayesServerAnalysis().HistogramDensityOptions()
    values = []
    for value in series.iteritems():
        v = bayesServerDiscovery().WeightedValue()
        if np.isnan(value[1]) or np.isinf(value[1]):
            continue
        v.setValue(jp.java.lang.Double(value[1]))
        v.setWeight(1.0)
        values.append(v)

    return bayesServerAnalysis().HistogramDensity.learn(jp.java.util.Arrays.asList(values), hdo)

def as_probability(series, output_column='cdf'):
    hist = create_histogram(series)
    df = pd.DataFrame(series)
    df[output_column] = series.apply(lambda x: hist.cdf(x))
    return df
