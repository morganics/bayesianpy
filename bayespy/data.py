import numpy as np

class DataFrameReader:
    def __init__(self, df):
        self._df = df
        self._row = None
        self._columns = df.columns.tolist()
        self.reset()

    def read(self):
        self._row = next(self._iterator, None)
        return self._row is not None

    def reset(self):
        self._iterator = self._df.itertuples()

    def __getitem__(self, key):
        # the df index of the row is at index 0
        ix = self._columns.index(key) + 1
        return self._row[ix]

class AutoType:
    def __init__(self, df):
        self._df = df

    def get_continuous_variables(self):
        cols = self._df.dtypes[(self._df.dtypes != "object") & (self._df.dtypes != "bool")].index.tolist()
        for col in cols:
            if len(self._df[col].unique()) > 20:
                yield col

    def get_discrete_variables(self):
        continuous = set(self.get_continuous_variables())
        for col in self._df.columns.tolist():
            if col not in continuous:
                yield col

class DataFrame:
    def is_float(dtype):
        return str(dtype) in {"float32", "float64"}

    def is_int(dtype):
        return str(dtype) in {"int32", "int64"}

    def is_bool(dtype):
        return str(dtype) == "bool"

    def is_string(dtype):
        return str(dtype) == "object" or str(dtype) == "O"

    def cast(df, col_name, value):
        if DataFrame.is_int(df[col_name].dtype):
            return int(value)
        if DataFrame.is_bool(df[col_name].dtype):
            return value == True
        if DataFrame.is_float(df[col_name].dtype):
            return float(value)

        return value


def coerce_float_to_int(df):
    for c in df.columns:
        if not DataFrame.is_float(df[c].dtype):
            continue

        can_convert = True
        for value in df[c].unique():
            if np.isnan(value) or int(value) != value: #no nan for int cols it seems!
                can_convert = False
                break

        if can_convert:
            df[c] = df[c].astype(int)

class Filter:
    def remove_static_variables(df):
        column_names = df.apply(lambda x: len(x.unique()) > 1)
        return df[column_names[column_names == True].index.tolist()]

    def remove_variable_variables(df):
        column_names = df.apply(lambda x: len(x.unique()) != len(df))
        return df[column_names[column_names == True].index.tolist()]

    def remove_discrete_variables_with_too_many_states(df, num_states = 30):
        column_names = df.select_dtypes(include=['object']).apply(lambda x: len(x.unique()) >= num_states)
        cols = list(set(df.columns.tolist()) - set(column_names[column_names == True].index.tolist()))
        return df[cols]

    def apply(df):
        return Filter.remove_discrete_variables_with_too_many_states(Filter.remove_variable_variables(Filter.remove_static_variables(df)))