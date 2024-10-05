import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error as MAE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
dane = pd.read_csv('D:/projkety/Building Energy Efficiency.csv')

# Display the first few rows of the data
print(dane.head())
print(dane.shape)
print(dane.isnull().sum())

# Plot histograms for each feature
dane.hist(bins=20, figsize=(20, 15))
plt.show()

# Scatter plot for Surface Area vs Cooling Load
import plotly.express as px
wykres = px.scatter(dane, x='Cooling Load', y='Surface Area', marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")
wykres.show()

# Scatter plot for Overall Height vs Heating Load
wykres = px.scatter(dane, x='Heating Load', y='Overall Height', marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")
wykres.show()

# Plot heatmap for correlation matrix
sns.set(style='whitegrid')
plt.figure(figsize=(12, 7))
mask = np.zeros_like(dane.corr(), dtype=bool)  # Changed np.bool to bool
mask[np.triu_indices_from(mask)] = True
sns.heatmap(dane.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, mask=mask, center=0)
plt.title("[Correlation heatmap]", fontsize=25)
plt.show()

# Prepare data
X = dane[['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']]
Y = dane[['Heating Load', 'Cooling Load']]
Y1 = dane[['Heating Load']]
Y2 = dane[['Cooling Load']]

# Split data into training and test sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, Y1, Y2, test_size=0.33, random_state=20)

# Normalize data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# Model evaluation
wyniki = pd.DataFrame(columns=['model', 'train_Heating', 'test_Heating', 'train_Cooling', 'test_Cooling', 'MAE_train_Heating', 'MAE_test_Heating', 'MAE_train_Cooling', 'MAE_test_Cooling'])

# List of models to evaluate
modele = [
    ['SVR', SVR()],
    ['DecisionTreeRegressor', DecisionTreeRegressor()],
    ['KNeighborsRegressor', KNeighborsRegressor()],
    ['RandomForestRegressor', RandomForestRegressor()],
    ['MLPRegressor', MLPRegressor()],
    ['AdaBoostRegressor', AdaBoostRegressor()],
    ['GradientBoostingRegressor', GradientBoostingRegressor()]
]

# Evaluate each model
for nazwa, model in modele:
    model.fit(X_train, y1_train)
    y1_train_pred = model.predict(X_train)
    y1_test_pred = model.predict(X_test)
    r2_train_heating = r2_score(y1_train, y1_train_pred)
    r2_test_heating = r2_score(y1_test, y1_test_pred)
    mae_train_heating = MAE(y1_train, y1_train_pred)
    mae_test_heating = MAE(y1_test, y1_test_pred)
    
    model.fit(X_train, y2_train)
    y2_train_pred = model.predict(X_train)
    y2_test_pred = model.predict(X_test)
    r2_train_cooling = r2_score(y2_train, y2_train_pred)
    r2_test_cooling = r2_score(y2_test, y2_test_pred)
    mae_train_cooling = MAE(y2_train, y2_train_pred)
    mae_test_cooling = MAE(y2_test, y2_test_pred)
    
    wyniki = pd.concat([wyniki, pd.DataFrame({'model': [nazwa], 
                                             'train_Heating': [r2_train_heating], 
                                             'test_Heating': [r2_test_heating], 
                                             'train_Cooling': [r2_train_cooling], 
                                             'test_Cooling': [r2_test_cooling], 
                                             'MAE_train_Heating': [mae_train_heating], 
                                             'MAE_test_Heating': [mae_test_heating], 
                                             'MAE_train_Cooling': [mae_train_cooling], 
                                             'MAE_test_Cooling': [mae_test_cooling]})], 
                      ignore_index=True)

print(wyniki.sort_values(by='test_Cooling', ascending=False))

# Hyperparameter tuning for DecisionTreeRegressor
dtr = DecisionTreeRegressor()
param_grid = {
    "criterion": ["squared_error", "absolute_error"],
    "min_samples_split": [14, 15, 16, 17],
    "max_depth": [5, 6, 7],
    "min_samples_leaf": [4, 5, 6],
    "max_leaf_nodes": [29, 30, 31, 32],
}

grid_cv_dtr = GridSearchCV(dtr, param_grid, cv=5)
grid_cv_dtr.fit(X_train, y2_train)
print("Best R-Squared for DecisionTreeRegressor: {}".format(grid_cv_dtr.best_score_))
print("Best parameters: \n{}".format(grid_cv_dtr.best_params_))

dtr = DecisionTreeRegressor(**grid_cv_dtr.best_params_)
dtr.fit(X_train, y1_train)
print("R-Squared DecisionTreeRegressor on training set={}".format(dtr.score(X_test, y1_test)))

dtr.fit(X_train, y2_train)
print("R-Squared DecisionTreeRegressor on test set={}".format(dtr.score(X_test, y2_test)))

# Hyperparameter tuning for RandomForestRegressor
param_grid = [{'n_estimators': [350, 400, 450], 'max_features': [1, 2], 'max_depth': [85, 90, 95]}]
rfr = RandomForestRegressor(n_jobs=-1)
grid_search_rfr = GridSearchCV(rfr, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_rfr.fit(X_train, y2_train)

print("Best R-Squared for RandomForestRegressor: {}".format(grid_search_rfr.best_score_))
print("Best parameters: \n{}".format(grid_search_rfr.best_params_))

rfr = RandomForestRegressor(**grid_search_rfr.best_params_)
rfr.fit(X_train, y1_train)
print("R-Squared RandomForestRegressor on training set={}".format(rfr.score(X_test, y1_test)))

rfr.fit(X_train, y2_train)
print("R-Squared RandomForestRegressor on test set={}".format(rfr.score(X_test, y2_test)))

# Hyperparameter tuning for GradientBoostingRegressor
param_grid = [{
    "learning_rate": [0.01, 0.02, 0.1],
    "n_estimators": [150, 200, 250],
    "max_depth": [4, 5, 6],
    "min_samples_split": [1, 2, 3],
    "min_samples_leaf": [2, 3],
    "subsample": [0.8, 1.0]
}]
gbr = GradientBoostingRegressor()
grid_search_gbr = GridSearchCV(gbr, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_gbr.fit(X_train, y2_train)

print("Best R-Squared dla GradientBoostingRegressor: {}".format(grid_search_gbr.best_score_))
print("Best parameters: \n{}".format(grid_search_gbr.best_params_))

gbr = GradientBoostingRegressor(**grid_search_gbr.best_params_)
gbr.fit(X_train, y1_train)
print("R-Squared GradientBoostingRegressor on training set={}".format(gbr.score(X_test, y1_test)))

gbr.fit(X_train, y2_train)
print("R-Squared GradientBoostingRegressor on test set={}".format(gbr.score(X_test, y2_test)))

# Hyperparameter tuning for CatBoostRegressor
model_cbr = CatBoostRegressor()
parametry = {
    'depth': [8, 10],
    'iterations': [10000],
    'learning_rate': [0.02, 0.03],
    'border_count': [5],
    'random_state': [42, 45]
}

grid = GridSearchCV(estimator=model_cbr, param_grid=parametry, cv=2, n_jobs=-1)
grid.fit(X_train, y2_train)
print("Best R-Squared dla CatBoostRegressor: {}".format(grid.best_score_))
print("Best estimator: \n{}".format(grid.best_estimator_))
print("Best parameters: \n{}".format(grid.best_params_))

model = CatBoostRegressor(**grid.best_params_)
model.fit(X_train, y1_train)
y1_pred = model.predict(X_test)

model.fit(X_train, y2_train)
y2_pred = model.predict(X_test)

r2_train_heating = r2_score(y1_train, model.predict(X_train))
r2_test_heating = r2_score(y1_test, model.predict(X_test))
r2_train_cooling = r2_score(y2_train, model.predict(X_train))
r2_test_cooling = r2_score(y2_test, model.predict(X_test))

mae_train_heating = MAE(y1_train, model.predict(X_train))
mae_test_heating = MAE(y1_test, model.predict(X_test))
mae_train_cooling = MAE(y2_train, model.predict(X_train))
mae_test_cooling = MAE(y2_test, model.predict(X_test))

print("CatBoostRegressor R-Squared on training set for heating load={}".format(r2_train_heating))
print("CatBoostRegressor R-Squared on test set for heating load={}".format(r2_test_heating))
print("CatBoostRegressor R-Squared on training set for cooling load={}".format(r2_train_cooling))
print("CatBoostRegressor R-Squared on test set for cooling load=={}".format(r2_test_cooling))

print("CatBoostRegressor MAE on training set for heating load={}".format(mae_train_heating))
print("CatBoostRegressor MAE on test set for heating load={}".format(mae_test_heating))
print("CatBoostRegressor MAE on training set for cooling load=={}".format(mae_train_cooling))
print("CatBoostRegressor MAE on test set for cooling load=={}".format(mae_test_cooling))

# Plot feature importances for CatBoost
importances = model.get_feature_importance()
nazwa_cech = X.columns
indeksy = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('feature importance', fontsize=20)
plt.barh(range(X.shape[1]), importances[indeksy], align='center', color='skyblue')
plt.yticks(range(X.shape[1]), [nazwa_cech[i] for i in indeksy])
plt.xlabel('feature importance')
plt.show()

# Plot actual vs predicted values
x_osi = range(len(y1_test))
plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
plt.plot(x_osi, y1_test, color='red', label="Real heating load")
plt.plot(x_osi, y1_pred, color='blue', label="Predicted heating load")
plt.title("Heating load: test data vs predicted data", fontsize=20)
plt.xlabel('X axis')
plt.ylabel('heating load (kW)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_osi, y2_test, color='green', label="Real cooling load")
plt.plot(x_osi, y2_pred, color='orange', label="Predicted cooling load")
plt.title("Cooling load: test data vs predicted data", fontsize=20)
plt.xlabel('Oś X')
plt.ylabel('Obciążenie chłodzące (kW)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.show()

# Define AAD function
def AAD(y_test, y_pred):
    return [(pred - actual) / actual * 100 for actual, pred in zip(y_test.values.flatten(), y_pred)]

# Plot AAD
plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
plt.plot(x_osi, AAD(y1_test, y1_pred), color='purple', label="Relative deviation of heating load")
plt.title("Heating Load - standard deviation", fontsize=20)
plt.xlabel('X Axis')
plt.ylabel('Error (%)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_osi, AAD(y2_test, y2_pred), color='magenta', label="Relative deviation of cooling load")
plt.title("Cooling Load - standard deviation", fontsize=20)
plt.xlabel('X Axis')
plt.ylabel('Error (%)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.show()
