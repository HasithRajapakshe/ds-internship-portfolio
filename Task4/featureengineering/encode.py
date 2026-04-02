import category_encoders as ce

# binary encoder use only garage


def binary_encoder(x_train, x_test,):
    b_encoder = ce.BinaryEncoder(cols=['Garage'])
    x_train_encoded = b_encoder.fit_transform(x_train)
    x_test_encoded = b_encoder.transform(x_test)
    return x_train_encoded, x_test_encoded


def frequency_encoder(x_train, x_test):
    f_encoder = ce.CountEncoder(cols=['Condition'])
    x_train_encoded = f_encoder.fit_transform(x_train)
    x_test_encoded = f_encoder.transform(x_test)
    return x_train_encoded, x_test_encoded


def target_encoder(x_train, x_test, y_train):
    t_encoder = ce.TargetEncoder(cols=['Location'])
    x_train_encoded = t_encoder.fit_transform(x_train, y_train)
    x_test_encoded = t_encoder.transform(x_test)
    return x_train_encoded, x_test_encoded
