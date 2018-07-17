import numpy as np
import matplotlib as mpl
from lib.fetch_data import FetchData
from lib.transformation_pipelines import TransformationPipelines
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def display_scores(model_name, scores):
    print(model_name)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

if __name__ == '__main__':
    fetch_data = FetchData()
    # housing_data = fetch_housing_data()
    housing = fetch_data.load_housing_data()
    housing.hist(bins=50, figsize=(20, 15))
    mpl.use('Agg')
    # housing.head()
    # train_set, test_set = split_train_test(housing, 0.2)
    # print len(train_set), "train +", len(test_set), "test"
    # train_set, test_set = train_test_split(housing,
    #                                        test_size=0.2,
    #                                        random_state=42)
    column_name = "income_cat"
    housing[column_name] = np.ceil(housing["median_income"] / 1.5)
    housing[column_name].where(housing[column_name] < 5, 5.0, inplace=True)
    train_set, test_set = fetch_data.stratified_split(housing, column_name)

    # housing_prop = fetch_data.get_data_proportion_by_column(housing, column_name)
    # train_prop = fetch_data.get_data_proportion_by_column(train_set, column_name)
    # test_prop = fetch_data.get_data_proportion_by_column(test_set, column_name)
    #
    # print 'housing_prop:', housing_prop
    # print 'train_prop:', train_prop
    # print 'test_prop:', test_prop

    for item in (train_set, test_set):
        item.drop([column_name], axis=1, inplace=True)

    # print train_set.columns, test_set.columns
    # train_data = train_set.copy()
    # train_data.plot(kind="scatter",
    #                 x="longitude",
    #                 y="latitude",
    #                 alpha=0.1,
    #                 figsize=(20, 15))

    # train_data.plot(kind="scatter",
    #                 x="longitude",
    #                 y="latitude",
    #                 alpha=0.4,
    #                 s=train_data["population"]/100,
    #                 label="population",
    #                 c="median_house_value",
    #                 cmap=plt.get_cmap("jet"),
    #                 colorbar=True)
    # plt.legend()

    # correlations
    # train_data["rooms_per_household"] = train_data["total_rooms"]/train_data["households"]
    # train_data["bedrooms_per_room"] = train_data["total_bedrooms"]/train_data["total_rooms"]
    # train_data["population_per_household"] = train_data["population"]/train_data["households"]
    #
    # corr_matrix = train_data.corr()
    # mhv_top_corr = corr_matrix["median_house_value"].sort_values(ascending=False)
    # print (mhv_top_corr)
    #
    # attributes = ["median_house_value", "median_income", "total_rooms",
    #               "housing_median_age"]
    # scatter_matrix(train_data[attributes], figsize=(12, 8))
    # plt.show()
    # plt.savefig('/vagrant/data_images/scatter_matrix.png', format='png')

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    # imputer = Imputer(strategy="median")
    # housing_num = housing.drop("ocean_proximity", axis=1)
    # imputer.fit(housing_num)
    # print(imputer.statistics_)

    # encoder = LabelEncoder()
    # housing_cat = housing["ocean_proximity"]
    # housing_cat_encoded = encoder.fit_transform(housing_cat)
    # print(housing_cat_encoded)
    # print(encoder.classes_)

    # encoder = OneHotEncoder()
    # housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
    # print(housing_cat_1hot.toarray())

    # encoder = LabelBinarizer()
    # housing_cat_1hot = encoder.fit_transform(housing_cat)
    # print(housing_cat_1hot)
    #
    # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(housing.values)
    # print(housing_extra_attribs)

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    pipeline = TransformationPipelines(num_attribs, cat_attribs)
    full_pipeline = pipeline.get_full_pipeline()
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared.shape)

    # lin_reg = LinearRegression()
    # lin_reg.fit(housing_prepared, housing_labels)

    # test_data = housing.iloc[:5]
    # test_labels = housing_labels.iloc[:5]
    # test_data_prepared = full_pipeline.transform(test_data)
    # print(test_data_prepared.shape)
    # print("Predictions:\t", lin_reg.predict(housing_prepared[:5]))
    # print("Labels:\t\t", list(housing_labels[:5]))

    # housing_predictions = lin_reg.predict(housing_prepared)
    # lin_mse = mean_squared_error(housing_labels, housing_predictions)
    # lin_rmse = np.sqrt(lin_mse)
    # lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
    #                              scoring="neg_mean_squared_error", cv=10)
    # lin_rmse_scores = np.sqrt(-lin_scores)
    # display_scores('linear', lin_rmse_scores)
    #
    # tree_reg = DecisionTreeRegressor()
    # tree_reg.fit(housing_prepared, housing_labels)
    # housing_predictions = tree_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_labels, housing_predictions)
    # tree_rmse = np.sqrt(tree_mse)
    # tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
    #                               scoring="neg_mean_squared_error", cv=10)
    # tree_rmse_scores = np.sqrt(-tree_scores)
    # display_scores('tree', tree_rmse_scores)

    #max_depth insure thie random forest regressor will not be killed
    forest_reg = RandomForestRegressor(max_depth=10, random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    # print(forest_reg)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores('forest', forest_rmse_scores)
