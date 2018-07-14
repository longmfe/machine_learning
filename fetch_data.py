import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
# from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
# import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


class FetchData():
    def fetch_housing_data(self,
                           housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)
            tgz_path = os.path.join(housing_path, "housing.tgz")
            urllib.request.urlretrieve(housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=housing_path)
            housing_tgz.close()

    def load_housing_data(self, housing_path=HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def split_train_test(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def test_set_check(self, identifier, test_ratio, hash):
        return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    def split_train_test_by_id(self, data, test_ratio,
                               id_column, hash=hashlib.md5):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_:
                                self.test_set_check(id_, test_ratio, hash))
        return data.loc[~in_test_set], data.loc[in_test_set]

    def stratified_split(self, data, column):
        split = StratifiedShuffleSplit(n_splits=1,
                                       test_size=0.2,
                                       random_state=42)
        for train_index, test_index in split.split(data, data[column]):
            strat_train_set = data.loc[train_index]
            strat_test_set = data.loc[test_index]
        return strat_train_set, strat_test_set

    def get_data_proportion_by_column(self, data, column):
        data_len = len(data)
        result = data[column].value_counts() / data_len
        return result
