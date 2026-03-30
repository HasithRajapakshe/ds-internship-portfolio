"""Evaluation: classification report for trained models."""

from sklearn.metrics import classification_report


def evaluate_model(model, x_test, y_test, model_name="Model"):
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(f"Classification Report — {model_name}:\n{report}")
    return report
