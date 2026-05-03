"""Main pipeline"""

import preprocess
import eda
import traintest
import evaluation


def main():

    df = preprocess.run()

    eda.run(df)

    x_train, x_test, y_train, y_test, trained_models = traintest.run(df)

    evaluation.run(trained_models, x_test, y_test, x_train, y_train)


if __name__ == '__main__':
    main()
