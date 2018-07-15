from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


class CustomerLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)
