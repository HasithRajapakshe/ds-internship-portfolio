"""Grid search helpers for each model type."""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def _run_grid_search(model_instance, param_grid, x_train, y_train):
    # cv=5 means dataset split into 5 flods
    grid_search = GridSearchCV(model_instance, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("Best params :", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)
    return grid_search.best_params_, grid_search.best_score_


def grid_search_model(x_train, y_train):
    param_grid = {
        # use regularization for prevent the overfiting,and control the model complexity
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
    }
    return _run_grid_search(LogisticRegression(max_iter=1000), param_grid, x_train, y_train)


def grid_search_model2(x_train, y_train):

    param_grid = {
        'max_depth': [None, 10, 20],  # levels
        'min_samples_split': [2, 5, 10],  # number of samples to split
        'min_samples_leaf': [1, 2, 4],  # number of samples at leaf
    }
    return _run_grid_search(DecisionTreeClassifier(), param_grid, x_train, y_train)


def grid_search_model3(x_train, y_train):

    param_grid = {
        'n_estimators': [50, 100, 200],  # numbers of trees
        'max_depth': [None, 10, 20],  # ech tree stop at level
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    return _run_grid_search(RandomForestClassifier(), param_grid, x_train, y_train)
# collections of decisions tree working together
