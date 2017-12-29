from typing import Iterable
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import numpy as np


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

def slowly_create_increasing_index(ddf:dd.DataFrame) -> dd.DataFrame:
    ddf['cs'] = 1
    ddf['cs'] = ddf.cs.cumsum()
    return ddf.set_index('cs')

def create_increasing_index(ddf:dd.DataFrame) -> dd.DataFrame:
    mps = int(len(ddf) / ddf.npartitions + 1000)
    values = ddf.index.values

    def do(x, max_partition_size, block_id=None):
        length = len(x)
        if length == 0:
            raise ValueError("Does not work with empty partitions. Consider using dask.repartition.")

        start = block_id[0] * max_partition_size
        return da.arange(start, start+length, chunks=1)

    series = values.map_blocks(do, max_partition_size=mps, dtype=np.int64)
    ddf2 = dd.concat([ddf, dd.from_array(series)], axis=1)
    return ddf


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
