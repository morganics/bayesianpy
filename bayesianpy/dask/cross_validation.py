import dask.array as da
import numpy as np
import numbers
import dask.dataframe as dd
import bayesianpy.utils.list

class KFold:

    def __init__(self, n_splits):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        #if not isinstance(shuffle, bool):
        #    raise TypeError("shuffle must be True or False;"
        #                    " got {0}".format(shuffle))

        self.n_splits = n_splits
        #self.shuffle = shuffle
        #self.random_state = random_state

    def split(self, ddf):
        partitions = ddf.random_split((1 / self.n_splits) * np.ones(self.n_splits, dtype=np.int))
        for split in range(self.n_splits):
            training_partitions = bayesianpy.utils.list.exclude(partitions, split)
            yield (dd.concat(training_partitions, axis=0, interleave_partitions=True), partitions[split])


    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

