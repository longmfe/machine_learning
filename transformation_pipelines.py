from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from combined_attributes_adder import CombinedAttributesAdder
from dataframe_selector import DataFrameSelector
from customer_label_binarizer import CustomerLabelBinarizer

# num_attribs = list(housing_num)
# cat_attribs = ["ocean_proximity"]


class TransformationPipelines():
    def __init__(self, num_attribs, cat_attribs):
        self.num_attribs = num_attribs
        self.cat_attribs = cat_attribs

    def get_full_pipeline(self):
        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.num_attribs)),
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.cat_attribs)),
            ('label_binarizer', CustomerLabelBinarizer())
        ])

        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ])

        return full_pipeline
