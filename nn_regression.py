from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    # mse = 0 # TODO Calculate MSE using mean_squared_error from sklearn.metrics (alrady imported)
    mse = mean_squared_error(targets, predictions)
    return mse


def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return: 
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    nn = MLPRegressor(hidden_layer_sizes=(100,), activation="tanh", alpha=0.01,
                      early_stopping=False, max_iter=500, solver="adam")
    nn.fit(X_train, y_train)

    # Calculate predictions
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')
    plt.scatter(np.array([i for  i in range(len(nn.loss_curve_))]), nn.loss_curve_)
    plt.show()

def grid_search(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=33)

    params = {"hidden_layer_sizes": [(10,), (20,), (50,), (100,)],
            "max_iter": [200, 500],
            "activation": ["logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "early_stopping": [True, False],
            "alpha": [0.0, 0.01, 0.1, 1.0]}

    regressor = MLPRegressor()

    grid = GridSearchCV(estimator=regressor, param_grid=params,
                        scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("---------------------")
    print("Best estimator: ")
    print(grid.best_estimator_)
    print("---------------------")
    print("Best parameters: ")
    print(grid.best_params_)
    print("---------------------")

    return grid.best_estimator_
