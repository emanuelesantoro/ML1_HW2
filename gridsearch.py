from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

features = np.load('data/x_datapoints.npy')
targets = np.load('data/y_datapoints.npy')


def calculate_mse(targets, predictions):
    mse = mean_squared_error(targets, predictions)
    return mse

# try out:
# - different numbers of neurons
# - different optimizers
# - some form of regularization

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=33)

all_params = {"hidden_layer_sizes": [(20,), (20,10), (100,)],
          "max_iter": [1000, 2000],
          "activation": ["logistic", "identity", "tanh", "relu"],
          "solver": ["lbfgs", "sgd", "adam"],
          "early_stopping": ["True", "False"],
          "alpha": [0.0, 0.01, 0.1, 1.0]}


params = {"hidden_layer_sizes": [(20,), (20, 10), (100,)],
          "max_iter": [200, 500],
          "activation": ["logistic", "tanh", "relu"],
          "solver": ["lbfgs", "sgd", "adam"],
          "early_stopping": [True, False],
          "alpha": [0.0, 0.01, 0.1, 1.0]}

regressor = MLPRegressor()

grid = GridSearchCV(estimator=regressor, param_grid=params,
                    scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)

print("---------------------")
print("Best estimator: ")
print(grid.best_estimator_)
print("---------------------")
print("Best parameters: ")
print(grid.best_params_)
print("---------------------")


nn = grid.best_estimator_
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)
print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')
