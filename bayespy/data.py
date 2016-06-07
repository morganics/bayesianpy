class AutoType:
    def __init__(self, df):
        self._df = df

    def get_continuous_variables(self):
        cols = self._df.dtypes[(self._df.dtypes != "object") & (self._df.dtypes != "bool")].index.tolist()
        for col in cols:
            if len(self._df[col].unique()) > 15:
                yield col

    def get_discrete_variables(self):
        continuous = set(self.get_continuous_variables())
        for col in self._df.columns.tolist():
            if col not in continuous:
                yield col

class DataFrame:
    def is_int(dtype):
        return str(dtype) in {"int32", "int64"}

    def is_bool(dtype):
        return str(dtype) == "bool"

    def is_string(dtype):
        return str(dtype) == "object" or str(dtype) == "O"