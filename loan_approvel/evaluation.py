"""Evaluation"""
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def run(trained_models, x_test, y_test, x_train, y_train):
    reports = {}

    for name, model in trained_models.items():
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred)
        print(f"\n Classification Report — {name}:\n{report}")
        reports[name] = report

    grid_results = {}

    print("\n[Logistic Regression]")
    grid_results['Logistic Regression'] = grid_search_logistic(
        x_train, y_train)

    print("\n[Decision Tree]")
    grid_results['Decision Tree'] = grid_search_decision_tree(
        x_train, y_train)

    print("\n[Random Forest]")
    grid_results['Random Forest'] = grid_search_random_forest(
        x_train, y_train)

    return reports, grid_results


def _run_grid_search(model_instance, param_grid, x_train, y_train):
    grid_search = GridSearchCV(
        model_instance, param_grid, cv=5, n_jobs=-1)  # flod into 5
    grid_search.fit(x_train, y_train)
    print("  Best params :", grid_search.best_params_)
    print("  Best CV score:", grid_search.best_score_)
    return grid_search.best_params_, grid_search.best_score_


def grid_search_logistic(x_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],  # find the best weights
    }
    return _run_grid_search(LogisticRegression(max_iter=1000), param_grid, x_train, y_train)


def grid_search_decision_tree(x_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],  # amount to split
        'min_samples_leaf': [1, 2, 4],  # last node amout data
    }
    return _run_grid_search(DecisionTreeClassifier(), param_grid, x_train, y_train)


def grid_search_random_forest(x_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],  # tress amount
        'max_depth': [None, 10, 20],  # levels
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    return _run_grid_search(RandomForestClassifier(), param_grid, x_train, y_train)
