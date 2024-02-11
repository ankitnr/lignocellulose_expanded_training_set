import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

spectra_data = pd.read_csv('all_spectra.csv')
composition_data = pd.read_csv('composition.csv')
#print(spectra_data.head())
#print(composition_data.head())

#Extract 'Sample' column for common identifier
commonIdentifier = composition_data.columns.tolist()
#print(commonIdentifier)

np.random.seed(12)

#Extract predictors X and response variable y
X = spectra_data.iloc[:,1:].values.T

y_cellulose = composition_data.iloc[0,1:].values.T
y_hemicellulose = composition_data.iloc[1,1:].values.T
y_lignin = composition_data.iloc[2,1:].values.T
y_fungi = composition_data.iloc[3,1:].values.T

#Split the data into train, test, and cross-validation sets
cv_ratio = 0.1  #fraction of data for cross validation
test_ratio = 0.2   #fraction of data for testing

#Randomly shuffle the data
idx = np.random.permutation(X.shape[0])
X_shuffled = X[idx,:]
y_cellulose_shuffled = y_cellulose[idx]
y_hemicellulose_shuffled= y_hemicellulose[idx]
y_lignin_shuffled = y_lignin[idx]
y_fungi_shuffled = y_fungi[idx]

X_train, X_test, y_cellulose_train, y_cellulose_test = train_test_split(X_shuffled, y_cellulose_shuffled, test_size= test_ratio, random_state= 123)
X_cellulose_train, X_cellulose_cv, y_cellulose_train, y_cellulose_cv = train_test_split(X_train, y_cellulose_train, test_size = cv_ratio/ (1-test_ratio), random_state= 123)

X_train, X_test, y_hemicellulose_train, y_hemicellulose_test = train_test_split(X_shuffled, y_hemicellulose_shuffled, test_size= test_ratio, random_state= 123)
X_hemicellulose_train, X_hemicellulose_cv, y_hemicellulose_train, y_hemicellulose_cv = train_test_split(X_train, y_hemicellulose_train, test_size = cv_ratio/ (1-test_ratio), random_state= 123)

X_train, X_test, y_lignin_train, y_lignin_test = train_test_split(X_shuffled, y_lignin_shuffled, test_size= test_ratio, random_state= 123)
X_lignin_train, X_lignin_cv, y_lignin_train, y_lignin_cv = train_test_split(X_train, y_lignin_train, test_size = cv_ratio/ (1-test_ratio), random_state= 123)

X_train, X_test, y_fungi_train, y_fungi_test = train_test_split(X_shuffled, y_fungi_shuffled, test_size= test_ratio, random_state= 123)
X_fungi_train, X_fungi_cv, y_fungi_train, y_fungi_cv = train_test_split(X_train, y_fungi_train, test_size = cv_ratio/ (1-test_ratio), random_state= 123)


# Define models
models_cellulose= {
    'Linear Regression': (LinearRegression(), {}),
    'Support Vector Machine': (SVR(), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'auto']}),
    'Decision Tree Regressor': (DecisionTreeRegressor(), {'max_depth': [100, 200, 300], 'min_samples_split': [2, 5, 10]}),
    'Partial Least Squares': (PLSRegression(), {'n_components': [2, 4, 6, 8], 'scale': [True, False]}),
    'Kernel Ridge Regression': (KernelRidge(), {'alpha': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']})
}

models_hemicellulose = {
    'Linear Regression': (LinearRegression(), {}),
    'Support Vector Machine': (SVR(), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'auto']}),
    'Decision Tree Regressor': (DecisionTreeRegressor(), {'max_depth': [2, 3, 5], 'min_samples_split': [2, 3]}),
    'Partial Least Squares': (PLSRegression(), {'n_components': [2, 4, 6, 8], 'scale': [True, False]}),
    'Kernel Ridge Regression': (KernelRidge(), {'alpha': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']})
}

models_lignin = {
    'Linear Regression': (LinearRegression(), {}),
    'Support Vector Machine': (SVR(), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'auto']}),
    'Decision Tree Regressor': (DecisionTreeRegressor(), {'max_depth': [100, 200, 300], 'min_samples_split': [2, 5, 10]}),
    'Partial Least Squares': (PLSRegression(), {'n_components': [2, 5, 10], 'scale': [True, False]}),
    'Kernel Ridge Regression': (KernelRidge(), {'alpha': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']})
}

models_fungi = {
    'Linear Regression': (LinearRegression(), {}),
    'Support Vector Machine': (SVR(), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'auto']}),
    'Decision Tree Regressor': (DecisionTreeRegressor(), {'max_depth': [100, 200, 300], 'min_samples_split': [2, 5, 10]}),
    'Partial Least Squares': (PLSRegression(), {'n_components': [2, 5, 10], 'scale': [True, False]}),
    'Kernel Ridge Regression': (KernelRidge(), {'alpha': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']})
}


# Train models and perform hyperparameter tuning
print("Cellulose:")
for model_name, (model, param_grid) in models_cellulose.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_cellulose_train, y_cellulose_train)
    best_model = grid_search.best_estimator_

    # Evaluate on cross-validation set
    y_cv_pred = best_model.predict(X_cellulose_cv)
    mae = mean_absolute_error(y_cellulose_cv, y_cv_pred)
    print(f'{model_name} - Mean Absolute Error: {mae}')


# Train models and perform hyperparameter tuning
print("Hemicellulose:")
for model_name, (model, param_grid) in models_hemicellulose.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_hemicellulose_train, y_hemicellulose_train)
    best_model = grid_search.best_estimator_

    # Evaluate on cross-validation set
    y_cv_pred = best_model.predict(X_hemicellulose_cv)
    mae = mean_absolute_error(y_hemicellulose_cv, y_cv_pred)
    print(f'{model_name} - Mean Absolute Error: {mae}')

# Train models and perform hyperparameter tuning
print("Lignin:")
for model_name, (model, param_grid) in models_lignin.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_lignin_train, y_lignin_train)
    best_model = grid_search.best_estimator_

    # Evaluate on cross-validation set
    y_cv_pred = best_model.predict(X_lignin_cv)
    mae = mean_absolute_error(y_lignin_cv, y_cv_pred)
    print(f'{model_name} - Mean Absolute Error: {mae}')

# Train models and perform hyperparameter tuning
print("Fungi:")
for model_name, (model, param_grid) in models_fungi.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_fungi_train, y_fungi_train)
    best_model = grid_search.best_estimator_

    # Evaluate on cross-validation set
    y_cv_pred = best_model.predict(X_fungi_cv)
    mae = mean_absolute_error(y_fungi_cv, y_cv_pred)
    print(f'{model_name} - Mean Absolute Error: {mae}')










