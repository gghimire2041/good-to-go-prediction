import pandas as pd
import numpy as np

from src.g2g_model.data.data_generator import G2GDataGenerator
from src.g2g_model.preprocessing.preprocessor import G2GPreprocessor


def test_preprocessor_fit_transform_shapes():
    gen = G2GDataGenerator(random_state=1)
    df = gen.generate_data(n_samples=200)

    pre = G2GPreprocessor()
    X, feature_names = pre.fit_transform(df)
    y = pre.get_target(df)

    assert pre.is_fitted is True
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(df)
    assert len(feature_names) == X.shape[1]
    assert y.shape[0] == len(df)


def test_preprocessor_transform_consistency():
    gen = G2GDataGenerator(random_state=2)
    df_train = gen.generate_data(n_samples=100)
    df_test = gen.generate_data(n_samples=20)

    pre = G2GPreprocessor()
    X_train, feature_names_train = pre.fit_transform(df_train)
    X_test, feature_names_test = pre.transform(df_test)

    assert X_test.shape[1] == X_train.shape[1]
    assert feature_names_train == feature_names_test

