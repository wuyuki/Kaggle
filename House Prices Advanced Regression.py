import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

# import data
train_df = pd.read_csv('./input/train.csv', index_col=0)
test_df = pd.read_csv('./input/test.csv', index_col=0)

# preview data
print(train_df.head())
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
prices.hist()
plt.show()
y_train = np.log1p(train_df.pop('SalePrice')+1)

# combine train and test data
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.shape)

#feature engineering
# 1. categorical(one-hot encoding)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
all_dummy_df = pd.get_dummies(all_df)
print(all_dummy_df.head())
# 2. numerical
# 2.1 get missing values count
all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)
# 2.2 fill missing values
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
# 2.3 normalize numerical values
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

# split train and test data
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
print(dummy_train_df.shape, dummy_test_df.shape)

# build model
# 1. adjust hyperparameters
# 1.1 Ridge Regression (good for putting all variables into model)
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, dummy_train_df.values, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(alphas, test_scores)
plt.show()
# 1.2 Random Forest
max_features = [.1, .3, .5, .7]
test_scores = []
for max_feature in max_features:
    clf = RandomForestRegressor(n_estimators=100, max_features=max_feature)
    test_score = np.sqrt(-cross_val_score(clf, dummy_train_df.values, y_train, cv=10, scoring='neg_mean_squared_error'))
    print(test_score)
    test_scores.append(np.mean(test_score))
plt.plot(max_features, test_scores)
plt.show()
# 2. ensemble
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
# 2.1 bagging
estimators = [1, 10, 15, 20, 25, 30, 40]
test_scores = []
for estimator in estimators:
    clf = BaggingRegressor(estimator=ridge, n_estimators=estimator)
    test_score = np.sqrt(-cross_val_score(clf, dummy_train_df.values, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(estimators, test_scores)
plt.show()
# 2.2 boosting
estimators = [10, 15, 20, 25, 30, 35, 40, 45, 50]
test_scores = []
for estimator in estimators:
    clf = AdaBoostRegressor(estimator=ridge, n_estimators=estimator)
    test_score = np.sqrt(-cross_val_score(clf, dummy_train_df.values, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(estimators, test_scores)
plt.show()
# 2.3 XGBoost
params = [1, 2, 3, 4, 5, 6]
test_scores = []
for param in params:
    clf = XGBRegressor(max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, dummy_train_df.values, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.show()

# predict
#1. stacking
ridge.fit(dummy_train_df.values, y_train)
rf.fit(dummy_train_df.values, y_train)
y_ridge = np.expm1(ridge.predict(dummy_test_df.values))
y_rf = np.expm1(rf.predict(dummy_test_df.values))
y_stacking = (y_ridge + y_rf) / 2 #stacking
#2. bagging
bagging = BaggingRegressor(base_estimator=ridge, n_estimators=30)
bagging.fit(dummy_train_df.values, y_train)
y_bagging = np.expm1(bagging.predict(dummy_test_df.values))
#3. boosting
boosting = AdaBoostRegressor(base_estimator=ridge, n_estimators=10)
boosting.fit(dummy_train_df.values, y_train)
y_boosting = np.expm1(boosting.predict(dummy_test_df.values))
#4. XGBoost
xgb = XGBRegressor(max_depth=2)
xgb.fit(dummy_train_df.values, y_train)
y_xgb = np.expm1(xgb.predict(dummy_test_df.values))

#output
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 
                                    'SalePrice_Stacking': y_stacking,
                                    'SalePrice_Bagging': y_bagging,
                                    'SalePrice_Boosting': y_boosting,
                                    'SalePrice_XGBoost': y_xgb})
submission_df.to_csv('./output/submission.csv', index=False)