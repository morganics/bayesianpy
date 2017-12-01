from bayesianpy.jni import bayesServer
from bayesianpy.jni import jp
import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import List
import logging

class Creatable:
    def create(self):
        pass

class CreateSqlDataReaderCommand(Creatable):

    def __init__(self, connection_string, query_string):
        self._conn = connection_string
        self._query = query_string

    def create(self):
        data_reader_command = bayesServer().data.DatabaseDataReaderCommand(
            self._conn,
            self._query)

        return data_reader_command


class CreateReaderOptions(Creatable):

    def __init__(self, index, weight=None):
        self._index = index
        self._weight = weight

    def create(self):
        return bayesServer().data.ReaderOptions(self._index) if self._weight is None \
            else bayesServer().data.ReaderOptions(self._index, self._weight)



class CreateDataFrameReaderCommand(Creatable):
    def __init__(self, dataframe):
        self._df = dataframe

    def create(self):
        return jp.JProxy("com.bayesserver.data.DataReaderCommand",
                                        inst=PandasDataReaderCommand(self._df))

def _to_java_class(data_type):
    """
    Converts numpy data type to equivalent Java class
    :param data_type: the numpy data type
    :return: The Java Class
    """
    if data_type == np.int32:
        return jp.java.lang.Integer(0).getClass()  # .class not currently supported by jpype
    if data_type == np.int64:
        return jp.java.lang.Long(0).getClass()  # .class not currently supported by jpype
    if data_type == np.float32:
        return jp.java.lang.Float(0).getClass()  # .class not currently supported by jpype
    if data_type == np.float64:
        return jp.java.lang.Double(0.0).getClass()  # .class not currently supported by jpype
    if data_type == np.bool:
        return jp.java.lang.Boolean(False).getClass()  # .class not currently supported by jpype
    if data_type == np.object:
        return jp.java.lang.String().getClass()  # .class not currently supported by jpype

    raise ValueError('dtype [{}] not currently supported'.format(data_type))


class PandasDataReader:
    def __init__(self, df:dd.DataFrame, partition_order:List[int]=None):
        self._logger = logging.getLogger(__name__)
        self._df = df
        self._columns = ["ix"] + [str(col) for col in self._df.columns.tolist()]
        self._dtypes = [df.index.dtype] + df.dtypes.tolist()
        self._i = 0
        self._ordered_partitions = partition_order
        self._iterator = self._iterator()
        self._object_accessors = None

    def _iterator(self):
        if hasattr(self._df, 'npartitions'):
            # is a dask dataframe.
            for i in range(self._df.npartitions):
                if i not in self._ordered_partitions:
                    continue

                ordered_partition = self._ordered_partitions[i]
                self._logger.info("Partition {}".format(ordered_partition))
                df = self._df.get_partition(ordered_partition).compute()
                for row in df.itertuples():
                    yield row
        else:
            # is a pandas dataframe.
            for row in self._df.itertuples():
                yield row

    def _create_object_accessors(self):
        self._object_accessors = []
        for columnIndex, col in enumerate(self._columns):

            data_type = self._dtypes[columnIndex]

            if data_type == np.int32 or data_type == np.int64:
                self._object_accessors.append(self.getInt)
                continue
            if data_type == np.float32 or data_type == np.float64:
                self._object_accessors.append(self.getFloat)
                continue
            if data_type == np.bool:
                self._object_accessors.append(self.getBoolean)
                continue
            if data_type == np.object:
                self._object_accessors.append(self.getString)
                continue

            raise ValueError("Dtype {} not supported in Dask Data Reader".format(data_type))

    def read(self):
        if self._object_accessors is None:
            self._create_object_accessors()

        try:
            self.current_row = next(self._iterator)
            self._i += 1
            if self._i % 10000 == 0:
                self._logger.info("Read {} Rows".format(self._i))
            return jp.JBoolean(True)
        except StopIteration:
            return jp.JBoolean(False)

    def close(self):
        self._logger.info("Closed Dask DataReader (read {} rows)".format(self._i))

    def getBoolean(self, columnIndex):
        return bool(self.current_row[columnIndex])

    def getColumnCount(self):
        return len(self._df.columns)

    def getColumnIndex(self, columnName):
        return self._columns.index(columnName)

    def _get_column_name(self, index) -> str:
        return self._columns[index]

    def getColumnType(self, columnIndex):
        return _to_java_class(self._df[self._get_column_name(columnIndex)].dtype)

    def getDouble(self, columnIndex):
        return float(self.current_row[columnIndex])

    def getFloat(self, columnIndex):
        return float(self.current_row[columnIndex])

    def getInt(self, columnIndex):
        return int(self.current_row[columnIndex])

    def getLong(self, columnIndex):
        return int(self.current_row[columnIndex])

    def getObject(self, columnIndex):
        if self.isNull(columnIndex):
            return None
        return self._object_accessors[columnIndex](columnIndex)

    def getString(self, columnIndex):
        return str(self.current_row[columnIndex])

    def isNull(self, columnIndex):
        return pd.isnull(self.current_row[columnIndex])


class PandasDataReaderCommand:
    def __init__(self, df:dd.DataFrame):
        self._df = df
        self._logger = logging.getLogger(__name__)
        self._i = 0
        self._ordered_partitions = None

    def _order_partitions(self, df):
        ordering = {}
        for partition in range(df.npartitions):
            p = df.get_partition(partition).head(1)
            if p.empty:
                continue

            ordering.update({partition: int(p.index[0])})

        partitions = sorted(ordering, key=ordering.get)
        self._logger.info("Ordered Partitions: {}".format(partitions))
        return partitions

    def executeReader(self):
        self._i += 1
        self._logger.info("Creating Dask Data Reader (iteration: {})".format(self._i))

        if self._ordered_partitions is None and hasattr(self._df, 'npartitions'):
            # is a dask dataframe
            self._ordered_partitions = self._order_partitions(self._df)

        return jp.JProxy("com.bayesserver.data.DataReader",
                                 inst=PandasDataReader(self._df, self._ordered_partitions))
