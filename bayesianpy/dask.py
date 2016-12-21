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

def to_sql(df, table, engine):
    if hasattr(df, 'to_sql'):
        df.to_sql(table, engine, if_exists='replace', index_label='ix', index=True)
    else:
        for partition in _get_df_partitions(df):
            partition.to_sql(table, engine, if_exists='append', index_label='ix', index=True)
