import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
# from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib as mpl
# import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def stratified_split(data, column):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data[column]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    return strat_train_set, strat_test_set


def get_data_proportion_by_column(data, column):
    data_len = len(data)
    result = data[column].value_counts() / data_len
    return result


if __name__ == '__main__':
    # housing_data = fetch_housing_data()
    housing = load_housing_data()
    housing.hist(bins=50, figsize=(20, 15))
    mpl.use('Agg')
    housing.head()
    # train_set, test_set = split_train_test(housing, 0.2)
    # print len(train_set), "train +", len(test_set), "test"
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    column_name = "income_cat"
    housing[column_name] = np.ceil(housing["median_income"] / 1.5)
    housing[column_name].where(housing[column_name] < 5, 5.0, inplace=True)
    train_set, test_set = stratified_split(housing, column_name)

    housing_prop = get_data_proportion_by_column(housing, column_name)
    train_prop = get_data_proportion_by_column(train_set, column_name)
    test_prop = get_data_proportion_by_column(test_set, column_name)

    print 'housing_prop:', housing_prop
    print 'train_prop:', train_prop
    print 'test_prop:', test_prop

    for item in (train_set, test_set):
        item.drop([column_name], axis=1, inplace=True)

    print train_set.columns, test_set.columns

