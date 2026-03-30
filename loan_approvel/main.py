"""Main pipeline"""

import pandas as pd


import preprocess
import eda
import traintest
import evaluation
import bestone

SELECTED_COLUMNS = [
    'income', 'credit_score', 'loan_amount', 'loan_approved'
]


def main():

    df = pd.read_csv(
        r"C:\Users\hasit\OneDrive\Desktop\loan_approvel\loan_approval.csv",
        usecols=SELECTED_COLUMNS,
    )

    print(df.head())
    # convert into 0 and 1
    df['loan_approved'] = df['loan_approved'].astype(int)
    print(df.head())

    df = preprocess.preprocess_data(df)
    eda.run_eda(df)
    x_train, x_test, y_train, y_test = traintest.split_data(df)

    model_lr = traintest.train_model(x_train, y_train)
    model_dt = traintest.train_model2(x_train, y_train)
    model_rf = traintest.train_model3(x_train, y_train)

    evaluation.evaluate_model(model_lr, x_test, y_test,
                              model_name="Logistic Regression")
    evaluation.evaluate_model(model_dt, x_test, y_test,
                              model_name="Decision Tree")
    evaluation.evaluate_model(model_rf, x_test, y_test,
                              model_name="Random Forest")

    print("\nGrid Search: Logistic Regression")
    best_params_lr, best_score_lr = bestone.grid_search_model(
        x_train, y_train)
    print(f"  Best Params: {best_params_lr}, Best Score: {best_score_lr:.4f}")

    print("\nGrid Search: Decision Tree")
    best_params_dt, best_score_dt = bestone.grid_search_model2(
        x_train, y_train)
    print(f"  Best Params: {best_params_dt}, Best Score: {best_score_dt:.4f}")

    print("\nGrid Search: Random Forest")
    best_params_rf, best_score_rf = bestone.grid_search_model3(
        x_train, y_train)
    print(f"  Best Params: {best_params_rf}, Best Score: {best_score_rf:.4f}")


if __name__ == "__main__":
    main()
