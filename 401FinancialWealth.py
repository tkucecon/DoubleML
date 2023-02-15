# --------------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------------

# modules
import numpy as np
import pandas as pd
import doubleml as dml
from doubleml.datasets import fetch_401K
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# set up the graph
sns.set()
colors = sns.color_palette()
plt.rcParams['figure.figsize'] = 10., 7.5
sns.set(font_scale=1.5)
sns.set_style('whitegrid', {'axes.spines.top': False,
                            'axes.spines.bottom': False,
                            'axes.spines.left': False,
                            'axes.spines.right': False})

# import data
data = fetch_401K(return_type='DataFrame')

# Temporary fix for https://github.com/DoubleML/doubleml-docs/issues/45 / https://github.com/scikit-learn/scikit-learn/issues/21997
# Can be removed when scikit-learn version 1.2.0 is released
dtypes = data.dtypes
dtypes['nifa'] = 'float64'
dtypes['net_tfa'] = 'float64'
dtypes['tw'] = 'float64'
dtypes['inc'] = 'float64'
data = data.astype(dtypes)

# check the data
data.head()
data.describe()

# --------------------------------------------------------------------------------------------------------
# Plot the data
# --------------------------------------------------------------------------------------------------------

# eligibility 
data['e401'].value_counts().plot(kind = 'bar', color = colors)
plt.title('Eligibility, 401(k)')
plt.xlabel('e401')
_ = plt.ylabel('count')
plt.show()

# participation
data['p401'].value_counts().plot(kind = 'bar', color = colors)
plt.title('Participation, 401(k)')
plt.xlabel('p401')
_ = plt.ylabel('count')
plt.show()

# compare the financial wealth accroding to eligibility
_ = sns.displot(data, x = "net_tfa", hue = "e401", col = "e401",
                kind = "kde", fill = True)
plt.show()

# --------------------------------------------------------------------------------------------------------
# Prepare the base and flexible data
# --------------------------------------------------------------------------------------------------------

# naive comparison
# ... but not sure if this difference comes from the participation to 401k (because of the endogeneity)
data[['p401', 'net_tfa']].groupby('p401').mean().diff()
data[['e401', 'net_tfa']].groupby('e401').mean().diff()

# Set up basic model: Specify variables for data-backend
features_base = ['age', 'inc', 'educ', 'fsize', 'marr',
                 'twoearn', 'db', 'pira', 'hown']

# Initialize DoubleMLData (data-backend of DoubleML)
data_dml_base = dml.DoubleMLData(data,
                                 y_col='net_tfa',
                                 d_cols='e401',
                                 x_cols=features_base)
print(data_dml_base)

# Set up a model according to regression formula with polynomials
features = data.copy()[['marr', 'twoearn', 'db', 'pira', 'hown']]

# dictionary of the covariates and degrees
poly_dict = {'age': 2,
             'inc': 2,
             'educ': 2,
             'fsize': 2}

# repeat and contain the polynomial regressors
for key, degree in poly_dict.items():
    poly = PolynomialFeatures(degree, include_bias = False)
    data_transf = poly.fit_transform(data[[key]])
    x_cols = poly.get_feature_names([key])
    data_transf = pd.DataFrame(data_transf, columns=x_cols)
    features = pd.concat((features, data_transf),
                          axis=1, sort=False)

# model data (add y and d)
model_data = pd.concat((data.copy()[['net_tfa', 'e401']], features.copy()),
                        axis=1, sort=False)

# Initialize DoubleMLData (data-backend of DoubleML)
data_dml_flex = dml.DoubleMLData(model_data, y_col='net_tfa', d_cols='e401')

# --------------------------------------------------------------------------------------------------------
# Partially Linear Regression Model (PLR)
# --------------------------------------------------------------------------------------------------------

# LASSO---------------------------------------------------------------------------------------------------

# Initialize learners
Cs = 0.0001*np.logspace(0, 4, 10)
lasso = make_pipeline(StandardScaler(), LassoCV(cv = 5, max_iter = 10000))
lasso_class = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                                 Cs = Cs, max_iter=1000))

# set the seed
np.random.seed(123)

# Initialize DoubleMLPLR model
dml_plr_lasso = dml.DoubleMLPLR(data_dml_base,
                                ml_l = lasso,
                                ml_m = lasso_class,
                                n_folds = 3)

# fit the model and check the summary
dml_plr_lasso.fit(store_predictions = True)
dml_plr_lasso.summary

# Estimate the ATE in the flexible model with lasso
np.random.seed(123)
dml_plr_lasso = dml.DoubleMLPLR(data_dml_flex,
                                ml_l = lasso,
                                ml_m = lasso_class,
                                n_folds = 3)

# fit the model and check the summary
dml_plr_lasso.fit(store_predictions=True)
lasso_summary = dml_plr_lasso.summary
lasso_summary

# Random Forest-------------------------------------------------------------------------------------------

# create random forest regressor and classifier
randomForest = RandomForestRegressor(
    n_estimators=500, max_depth=7, max_features=3, min_samples_leaf=3)
randomForest_class = RandomForestClassifier(
    n_estimators=500, max_depth=5, max_features=4, min_samples_leaf=7)

# Estimate the ATE in the flexible model with random forest
np.random.seed(123)
dml_plr_forest = dml.DoubleMLPLR(data_dml_flex,
                                 ml_l = randomForest,
                                 ml_m = randomForest_class,
                                 n_folds = 3)

# fit the model and check the summary
dml_plr_forest.fit(store_predictions = True)
forest_summary = dml_plr_forest.summary
forest_summary

# Tree--------------------------------------------------------------------------------------------------

# create tree regressor and classifier
trees = DecisionTreeRegressor(
    max_depth=30, ccp_alpha=0.0047, min_samples_split=203, min_samples_leaf=67)
trees_class = DecisionTreeClassifier(
    max_depth=30, ccp_alpha=0.0042, min_samples_split=104, min_samples_leaf=34)

# Estimate the ATE in the flexible model with tree
np.random.seed(123)
dml_plr_tree = dml.DoubleMLPLR(data_dml_flex,
                               ml_l = trees,
                               ml_m = trees_class,
                               n_folds = 3)

# fit the model and check the summary
dml_plr_tree.fit(store_predictions=True)
tree_summary = dml_plr_tree.summary
tree_summary

# Boosted Tree------------------------------------------------------------------------------------------

# create boosted tree regressor and classifier
boost = XGBRegressor(n_jobs=1, objective = "reg:squarederror",
                     eta=0.1, n_estimators=35)
boost_class = XGBClassifier(use_label_encoder=False, n_jobs=1,
                            objective = "binary:logistic", eval_metric = "logloss",
                            eta=0.1, n_estimators=34)

# Estimate the ATE in the flexible model with the XGBoost
np.random.seed(123)
dml_plr_boost = dml.DoubleMLPLR(data_dml_flex,
                                ml_l = boost,
                                ml_m = boost_class,
                                n_folds = 3)

# fit the model and check the summary
dml_plr_boost.fit(store_predictions=True)
boost_summary = dml_plr_boost.summary
boost_summary

# Summary-----------------------------------------------------------------------------------------------

# summarise
plr_summary = pd.concat((lasso_summary, forest_summary, tree_summary, boost_summary))
plr_summary.index = ['lasso', 'forest', 'tree', 'xgboost']
plr_summary[['coef', '2.5 %', '97.5 %']]

# plot the result
errors = np.full((2, plr_summary.shape[0]), np.nan)
errors[0, :] = plr_summary['coef'] - plr_summary['2.5 %']
errors[1, :] = plr_summary['97.5 %'] - plr_summary['coef']
plt.errorbar(plr_summary.index, plr_summary.coef, fmt='o', yerr=errors)
plt.ylim([0, 12500])

plt.title('Partially Linear Regression Model (PLR)')
plt.xlabel('ML method')
_ =  plt.ylabel('Coefficients and 95%-CI')
plt.show()

