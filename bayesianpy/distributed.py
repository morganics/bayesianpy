from typing import Iterable
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import numpy as np
import pathos.multiprocessing as mp
import logging

class DaskPool:
    '''
    We need to use DaskPool to avoid hanging in linux when multiprocessing is required. Jpype hangs
    when processes are not spawned.
    '''
    def __init__(self, processes:int=None, mode='multiprocessing'):
        self._processes = processes
        self._logger = logging.getLogger(__name__)
        self._mode = mode
        self._pool = None
        self._options = None

    def _calc_threads(self):
        if mp.cpu_count() == 1:
            max = 1
        else:
            max = mp.cpu_count() - 1

        return max

    def __enter__(self):

        if self._mode == 'multiprocessing':
            import dask.multiprocessing
            import multiprocess.context as ctx

            ctx._force_start_method('spawn')


            processes = self._calc_threads() if self._processes is None else self._processes
            self._pool = mp.Pool(processes=processes)
            self._options = dask.context.set_options(pool=self._pool, get=dask.multiprocessing.get)
            self._logger.info("Starting processing with {} processes".format(processes))
        elif self._mode == 'synchronous':
            import dask.local
            self._options = dask.context.set_options(get=dask.local.get_sync)
            self._logger.info("Starting synchronous processing")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._options is not None:
            self._options.__exit__(exc_type, exc_val, exc_tb)

        if self._pool is not None:
            self._pool.__exit__(exc_type, exc_val, exc_tb)

        self._options = None
        self._pool = None

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
