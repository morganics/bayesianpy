from typing import Iterable
import pandas as pd

def compute(df):
    if hasattr(df, 'compute'):
        return df.compute()
    return df

def empty(df):
    if not hasattr(df, 'empty'):
        return all(p.empty for p in _get_df_partitions(df))
    else:
        return df.empty

def _get_df_partitions(df) -> Iterable[pd.DataFrame]:
    for partitition in range(0, df.npartitions):
        yield df.get_partition(partitition).compute()

def _is_pandas(df):
    return isinstance(df, pd.DataFrame)

def to_sql(df, table, engine, index_label='ix', index=True, if_exists=None):
    if hasattr(df, 'to_sql'):
        mode = 'replace'
        if if_exists is not None:
            mode = if_exists
        #a = "replace" if_exists is not None else if_exists
        df.to_sql(table, engine, if_exists=mode, index_label=index_label, index=index)
    else:
        mode = 'append'
        if if_exists is not None:
            mode = if_exists
        for partition in _get_df_partitions(df):
            partition.to_sql(table, engine, if_exists=mode, index_label=index_label, index=index)
