from typing import Dict, List

import numpy as np
import logging
import pandas as pd
from sqlalchemy import create_engine
import uuid
import shutil
from bayesianpy.jni import bayesServer, bayesServerAnalysis, bayesServerDiscovery, jp
import os
from bayesianpy.decorators import listify
import dask.dataframe as dd
from typing import Iterable
import bayesianpy.dask as dk
from collections import defaultdict
import bayesianpy.utils

class DataFrameReader:
    def __init__(self, df):
        self._df = df
        self._row = None
        self._columns = df.columns.tolist()
        self.reset()
        self.row_index = 0
        self._writer = DataFrameWriter(df)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def read(self) -> bool:
        self._row = next(self._iterator, None)
        self.row_index += 1
        return self._row is not None

    def columns(self):
        return self._columns

    def reset(self) -> None:
        self._iterator = self._df.itertuples()

    def get_index(self):
        return self._row[0]

    def index(self):
        return self._row[0]

    def to_dict(self, columns: List[str] = None):
        return self.row(columns=columns)

    def tolist(self, cols) -> List[object]:
        return [self.__getitem__(c) for c in cols]

    def row(self, columns: List[str] = None) -> Dict[str, object]:
        cols = set(self._columns if columns is None else columns)
        return {c : self.__getitem__(c) for c in self._columns if c in cols}

    def writer(self) -> 'DataFrameWriter':
        if self._row is not None:
            return self._writer.with_index(self.get_index())
        else:
            return self._writer

    def set_value(self, column, value):
        self.writer().set_value(column, value)

    def __getitem__(self, key) -> object:
        # the df index of the row is at index 0
        try:
            if type(key) is list:
                print(key)
                ix = [self._columns.index(key) + 1 for k in key]
            else:
                ix = self._columns.index(key) + 1
            return self._row[ix]
        except BaseException as e:
            return None

    def __next__(self) -> 'DataFrameReader':
        if self.read():
            return self
        else:
            raise StopIteration

    def __iter__(self) -> 'DataFrameReader':
        return self

class DataFrameWriter:

    def __init__(self, df):
        self._columns = {}
        self._row_indices = set()
        self._df = df

    def with_index(self, index):

        if not bayesianpy.data.DataFrame.is_int(self._df.index.dtype):
            raise IndexError("Index has to be of type int to use the Writer")

        if index not in self._row_indices:
            self._row_indices.add(index)

        self._current_row_index = index
        return self

    def set_value(self, column, value):
        if column not in self._columns:
            if isinstance(value, str):
                self._columns.update({
                    column: np.empty((1, len(self._df)), dtype="object")
                })
                self._columns[column][:] = ""
            elif isinstance(value, bool):
                self._columns.update({
                    column: np.empty((1, len(self._df)), dtype="bool")
                })
                self._columns[column][:] = False
            else:
                self._columns.update({
                    column: np.empty((1, len(self._df)))
                })
                self._columns[column][:] = np.NAN

        self._columns[column][0, self._current_row_index] = value

    def as_dataframe(self):
        self._columns['ix'] = np.array(self._df.index)
        df = pd.DataFrame({key: col.flatten() for key,col in self._columns.items()})
        return df.set_index('ix')

    def flush(self):
        self._df = self.get_dataframe()

    def get_dataframe(self, df: pd.DataFrame = None):
        df1 = self.as_dataframe()
        if df is None:
            return self._df.join(df1)
        else:
            return df.join(df1)

class PandasDataReader:
    def __init__(self, df:dd.DataFrame, partition_order:List[int]=None):
        self._logger = logging.getLogger(__name__)
        self._df = df
        self._columns = ["index"] + [str(col) for col in self._df.columns.tolist()]
        self._dtypes = [df.index.dtype] + df.dtypes.tolist()
        self._i = 0
        self._ordered_partitions = partition_order
        self._iterator = self._iterator()

    def _iterator(self):
        if hasattr(self._df, 'npartitions'):
            # is a dask dataframe.
            for i in range(self._df.npartitions):
                ordered_partition = self._ordered_partitions[i]
                self._logger.info("Partition {}".format(ordered_partition))
                df = self._df.get_partition(ordered_partition).compute()
                for row in df.itertuples():
                    yield row
        else:
            # is a pandas dataframe.
            for row in self._df.itertuples():
                yield row

    def read(self):
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
        return bayesianpy.data._to_java_class(self._df[self._get_column_name(columnIndex)].dtype)

    def getDouble(self, columnIndex):
        return float(self.current_row[columnIndex])

    def getFloat(self, columnIndex):
        return float(self.current_row[columnIndex])

    def getInt(self, columnIndex):
        return int(self.current_row[columnIndex])

    def getLong(self, columnIndex):
        return int(self.current_row[columnIndex])

    def getObject(self, columnIndex):

        data_type = self._dtypes[columnIndex]

        if data_type == np.int32:
            return self.getInt(columnIndex)
        if data_type == np.int64:
            return self.getInt(columnIndex)
        if data_type == np.float32:
            return self.getFloat(columnIndex)
        if data_type == np.float64:
            return self.getFloat(columnIndex)
        if data_type == np.bool:
            return self.getBoolean(columnIndex)
        if data_type == np.object:
            return self.getString(columnIndex)

        raise ValueError("Dtype {} not supported in Dask Data Reader".format(data_type))

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
            ordering.update({partition: int(df.get_partition(partition).head(1).index[0])})

        partitions = sorted(ordering, key=ordering.get)
        self._logger.info("Ordered Partitions: {}".format(partitions))
        return partitions

    def executeReader(self):
        self._i += 1
        self._logger.info("Creating Dask Data Reader (iteration: {})".format(self._i))

        if self._ordered_partitions is None and hasattr(self._df, 'npartitions'):
            # is a dask dataframe
            self._ordered_partitions = self._order_partitions(self._df)

        return bayesianpy.jni.jp.JProxy("com.bayesserver.data.DataReader",
                                 inst=PandasDataReader(self._df, self._ordered_partitions))

class AutoType:
    def __init__(self, df, discrete=[], continuous=[], continuous_to_discrete_limit = 20, max_states=150):
        self._df = df
        self._continuous_to_discrete_limit = continuous_to_discrete_limit
        self._continuous = continuous
        self._discrete = discrete
        self._max_states = max_states

    @listify
    def get_continuous_variables(self):
        cols = self._df.columns.tolist()
        for col in cols:
            try:
                if col in self._discrete:
                    continue
                elif col in self._continuous:
                    yield str(col)
                elif not DataFrame.is_float(self._df[str(col)].dtype) and not DataFrame.is_int(self._df[str(col)].dtype):
                    continue

                elif len(self._df[str(col)].unique()) > self._continuous_to_discrete_limit:
                    yield str(col)
            except BaseException as e:
                print(col, e)

    @listify
    def get_discrete_variables(self):
        continuous = set(self.get_continuous_variables())
        for col in self._df.columns.tolist():
            l = len(dk.compute(self._df[str(col)].unique()))
            if col in self._continuous:
                continue
            elif l > self._max_states or l <= 1:
                continue
            elif DataFrame.is_timestamp(self._df[str(col)].dtype):
                continue
            elif col in self._discrete:
                yield str(col)
            elif col not in continuous:
                yield str(col)


class DataFrame:

    def __init__(self, df):
        self._df = df

    @staticmethod
    def is_timestamp(dtype):
        for ts in ['timestamp64', 'timedelta64', 'datetime64']:
            if ts in str(dtype):
                return True

        return False

    @staticmethod
    def get_boolean_columns(df):
        return [col for col in df.columns if DataFrame.is_bool(df[col].dtype)]

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
    def coerce_to_numeric(df: pd.DataFrame, logger: logging.Logger, cutoff=0.10, ignore=[]) -> pd.DataFrame:
        for col in df.columns:
            if DataFrame.is_numeric(df[col].dtype):
                continue

            if DataFrame.is_timestamp(df[col].dtype):
                continue

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
    def coerce_to_boolean(df: pd.DataFrame, ignore=[]):
        for col in df.columns:
            if col in ignore:
                continue

            if df[col].min() == df[col].max():
                continue

            if len(df[col].dropna().unique()) != 2:
                continue

            values = df[col].dropna().unique()
            series = df[col].astype(bool)

            if len(df[df[col] == values[0]]) != len(series[series == True]) and \
                            len(df[df[col] == values[1]]) != len(series[series == True]):
                continue

            df[col] = series

        return df

    @staticmethod
    def cast2(dtype, value):
        if DataFrame.is_int(dtype):
            return int(value)
        if DataFrame.is_bool(dtype):
            return value == True or value == str(True)
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
            if col in df.columns:
                df.loc[df[col] == 0, col] = df[col].apply(lambda x: np.random.normal(0, 3))


class DaskDataFrame:

    def __init__(self, df: dd.DataFrame):
        self._df = df
        self.empty = all(p.empty for p in self._get_df_partitions())

    def __len__(self):
        return len(self._df)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return self._df.__getattribute__(item)

    def __getitem__(self, key):
        # the df index of the row is at index 0
        return self._df.__getitem__(key)

    def _get_df_partitions(self) -> Iterable[pd.DataFrame]:
        for partitition in range(0, self._df.npartitions):
            yield self._df.get_partition(partitition).compute()

    def to_sql(self, table, engine, index_label='ix', index=True):
        for partition in range(0, self._df.npartitions):
            df = self._df.get_partition(partition)
            df.compute().to_sql(table, engine, if_exists='append', index_label=index_label, index=True)



class Filter:
    @staticmethod
    def remove_static_variables(df: pd.DataFrame, cutoff=1, logger:logging.Logger=None):
        column_names = df.apply(lambda x: len(x.unique()) > cutoff)

        if logger is not None:
            logger.info("Removing {}".format(", ".join(df.columns[column_names == False])))

        return df[column_names[column_names == True].index.tolist()].copy()

    @staticmethod
    def remove_variable_variables(df: pd.DataFrame):
        column_names = df.apply(lambda x: len(x.unique()) != len(df))
        return df[column_names[column_names == True].index.tolist()]

    @staticmethod
    def remove_mostly_empty_variables(df: pd.DataFrame, cutoff=0.1):
        length = len(df)
        for column in df.columns.tolist():
            if len(df[column].dropna()) / length <= cutoff:
                df = df.drop(column, axis=1)

        return df

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


class DataSet:
    def __init__(self, df: pd.DataFrame, logger: logging.Logger=None,
                 identifier: str=None, weight_column: str=None,
                 ):

        if identifier is None:
            self.uuid = str(uuid.uuid4()).replace("-","")
        else:
            self.uuid = identifier

        self._logger = logger if logger is not None else logging.getLogger()
        self.data = df
        self._weight_column = weight_column

    def subset(self, indices:List[int]) -> 'DataSet':
        return DataSet(self.data.iloc[indices], self._logger, identifier=self.uuid)

    def get_reader_options(self):
        return bayesServer().data.ReaderOptions("ix") if self._weight_column is None \
            else bayesServer().data.ReaderOptions("ix", self._weight_column)

    def get_dataframe(self) -> pd.DataFrame:
        return self.data

    def write(self, if_exists:str=None):
        pass

    def create_data_reader_command(self):
        pass

    def cleanup(self):
        pass

    def __enter__(self):
        self.write()
        return self

    def __exit__(self, type, value, traceback):
        self.cleanup()

class DataTableDataSet(DataSet):

    ''' Not very good for large datasets '''
    def __init__(self, df: pd.DataFrame, logger: logging.Logger=None,
                 identifier: str=None, weight_column: str=None,
                 ):
        super().__init__(df, logger, identifier, weight_column)

    def write(self, if_exists:str=None):
        bayes = jp.JPackage("com.bayesserver")
        bayes_data = bayes.data

        data_table = bayes_data.DataTable()
        cols = data_table.getColumns()

        for name, data_type in self.data.dtypes.iteritems():
            java_class = _to_java_class(data_type)
            data_column = bayes_data.DataColumn(name, java_class)
            cols.add(data_column)

        for index, row in self.data.iterrows():
            data_table.getRows().add(row)

        self._data_table = data_table

    def create_data_reader_command(self):
        """
        Get the data reader
        :param indexes: training/ testing indexes
        :return: a a DatabaseDataReaderCommand
        """
        data_reader_command = bayesServer().data.DataTableDataReaderCommand(self._data_table)

        return data_reader_command


class SqlDataSet(DataSet):
    def __init__(self, df: pd.DataFrame, logger:logging.Logger=None, identifier=None, weight_column=None):
        super().__init__(df, logger, identifier=identifier, weight_column=weight_column)
        self._engine = None
        self.table = "table_" + self.uuid

    def get_connection(self):
        pass

    def create_data_reader_command(self):
        """
        Get the data reader
        :param indexes: training/ testing indexes
        :return: a a DatabaseDataReaderCommand
        """
        query = "select * from {} where ix in ({}) order by ix asc".format(self.table, ",".join(str(i) for i in self.data.index))

        data_reader_command = bayesServer().data.DatabaseDataReaderCommand(
            self.get_connection(),
            query)

        return data_reader_command

    def write(self, if_exists:str=None):
        self._logger.info("Writing rows to storage")
        dk.to_sql(self.data, self.table, self._engine, if_exists=if_exists)
        self._logger.info("Finished writing rows to storage")

class ExcelDataSet(DataSet):
    def __init__(self, df: pd.DataFrame, db_folder:str=None, logger:logging.Logger=None, identifier=None, weight_column=None):
        super().__init__(df, logger, identifier=identifier, weight_column=weight_column)
        self._engine = None
        self.table = "table_" + self.uuid

        self._db_dir = db_folder if db_folder is not None \
            else bayesianpy.utils.get_path_to_parent_dir(os.path.basename(os.getcwd()))

        self._create_folder()

    def get_connection(self):
        return \
            "jdbc:odbc:Driver={{Microsoft Excel Driver(*.xlsx)}};" \
                "DBQ={}.xlsx".format(os.path.join(self._db_dir, self.table))

    def _create_folder(self):
        if not os.path.exists(os.path.join(self._db_dir, "db")):
            os.makedirs(os.path.join(self._db_dir, "db"))

    def create_data_reader_command(self):
        """
        Get the data reader
        :param indexes: training/ testing indexes
        :return: a a DatabaseDataReaderCommand
        """
        query = "select * from [Sheet1$] where ix in ({}) order by ix asc".format(self.table, ",".join(str(i) for i in self.data.index))

        data_reader_command = bayesServer().data.DatabaseDataReaderCommand(
            self._get_connection(),
            query)

        return data_reader_command

    def write(self, if_exists:str=None):
        #from pyexcelerate import Workbook
        self._logger.info("Writing rows to storage")
        #wb = Workbook()
        #wb.new_sheet("Sheet1", data=self.data)
        #wb.save("{}.xlsx".format(os.path.join(self._db_dir, self.table)))
        writer = pd.ExcelWriter("{}.xlsx".format(os.path.join(self._db_dir, self.table))
                                , engine='xlsxwriter')
        self.data.to_excel(writer)
        writer.save()
        self._logger.info("Finished writing rows to storage")


class MysqlDataSet(SqlDataSet):
    ''' Good for larger datasets '''
    def __init__(self, df: pd.DataFrame, username:str, password:str, server:str,
                 logger: logging.Logger = None, identifier=None):
        super().__init__(df, logger, identifier=identifier)
        self._engine = self._create_mysql_engine(username, password, server, self.uuid)
        self._server = server
        self._username = username
        self._password = password

    def _create_mysql_engine(self, username, password, server, database):
        return create_engine('mysql://{}:{}@{}/{}'.format(username, password, server, database))

    def get_connection(self):
        return "jdbc:mysql://{}:{}@{}/{}".format(self._username, self._password, self._server, self.uuid)



    def subset(self, indices:List[int]) -> 'DataSet':
        return MysqlDataSet(self.data.iloc[indices], self._username, self._password,
                            self._server, identifier=self.uuid)

    def write(self, if_exists:str=None):
        jp.java.lang.Class.forName("com.mysql.jdbc.Driver",
                                   True, jp.java.lang.ClassLoader.getSystemClassLoader())
        from sqlalchemy_utils import database_exists, create_database
        if not database_exists(self._engine.url):
             create_database(self._engine.url)

        super().write(if_exists=if_exists)

class FirebirdDataSet(SqlDataSet):
    ''' Good for larger datasets '''
    def __init__(self, df: pd.DataFrame, username:str, password:str, server:str,
                 logger: logging.Logger = None, identifier=None, page_size:int=16384):
        if identifier is None:
            import time
            identifier = str(int(time.time()))

        super().__init__(df, logger, identifier=identifier)
        self._engine = self._create_firebird_engine(username, password, server, self.uuid)
        self._server = server
        self._username = username
        self._password = password
        self._pagesize = page_size

    def _dsn(self, username, password, server, database):
        return "{}:{}@{}:3050/{}".format(username, password, server, database)

    def _create_firebird_engine(self, username, password, server, database):
        import fdb
        return create_engine('firebird+fdb://{}'.format(self._dsn(username, password, server, database)))

    def get_connection(self):
        return "jdbc:firebirdsql://{}:3050/{}?userName={}&password={}&sqlDialect=3".format(self._server, self.uuid, self._username, self._password)

    def subset(self, indices:List[int]) -> 'DataSet':
        return FirebirdDataSet(self.data.iloc[indices], self._username, self._password,
                            self._server, identifier=self.uuid)

    def write(self, if_exists:str=None):

        def _create_database():
            return self._engine.dialect.dbapi.create_database(
                user=self._username, password=self._password, host=self._server, database=self.uuid,
                                                       page_size=self._pagesize
                                                       )

        from sqlalchemy_utils import database_exists
        from sqlalchemy import exc as dbexceptions
        try:
            if not database_exists(self._engine.url):
                a = _create_database()
        except dbexceptions.DatabaseError:
            a = _create_database()

        super().write(if_exists=if_exists)

class DefaultDataSet(SqlDataSet):

    def __init__(self, df: pd.DataFrame, db_folder:str=None, logger:logging.Logger=None, identifier=None):

        super().__init__(df, logger, identifier)

        self._db_dir = db_folder if db_folder is not None \
            else bayesianpy.utils.get_path_to_parent_dir(os.path.basename(os.getcwd()))

        self._create_folder()

        self._engine = self._create_sqlite_engine()

    def get_connection(self):
        return "jdbc:sqlite:{}.db".format(os.path.join(self._db_dir, "db", self.uuid))

    def _create_sqlite_engine(self):
        filename = "sqlite:///{}.db".format(os.path.join(self._db_dir, "db", self.uuid))
        return create_engine(filename)

    def _create_folder(self):
        if not os.path.exists(os.path.join(self._db_dir, "db")):
            os.makedirs(os.path.join(self._db_dir, "db"))

    def cleanup(self):
        self._logger.debug("Cleaning up: deleting db folder")
        try:
            shutil.rmtree(os.path.join(self._db_dir, "db"))
        except:
            self._logger.error("Could not delete the db folder {} for some reason.".format(self._db_dir))

    def subset(self, indices:List[int]) -> 'DataSet':
        return DefaultDataSet(self.data.iloc[indices], self._db_dir, self._logger, identifier=self.uuid)


class DaskDataset(DataSet):
    def __init__(self, df: dd.DataFrame):
        super().__init__(df)
        self._df = df

    def get_dataframe(self) -> dd.DataFrame:
        return self._df

    def create_data_reader_command(self):
        return bayesianpy.jni.jp.JProxy("com.bayesserver.data.DataReaderCommand",
                                                       inst=PandasDataReaderCommand(self._df))

    def cleanup(self):
        pass

    def subset(self, indices):
        raise NotImplementedError("Haven't got round to doing this yet.")


def as_probability(series, output_column='cdf'):
    hist = create_histogram(series)
    df = pd.DataFrame(series)
    df[output_column] = series.apply(lambda x: hist.cdf(x))
    return df
