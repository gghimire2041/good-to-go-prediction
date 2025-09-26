import numpy as np
from sklearn.model_selection import train_test_split

from src.g2g_model.data.data_generator import G2GDataGenerator
from src.g2g_model.preprocessing.preprocessor import G2GPreprocessor
from src.g2g_model.models.catboost_model import G2GCatBoostModel


def test_model_training_and_prediction_range():
    gen = G2GDataGenerator(random_state=3)
    df = gen.generate_data(n_samples=300)

    pre = G2GPreprocessor()
    X, feature_names = pre.fit_transform(df)
    y = pre.get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    config = {
        'hyperparameters': {
            'iterations': 50,
            'learning_rate': 0.15,
            'depth': 4,
            'verbose': False,
            'early_stopping_rounds': 20,
            'loss_function': 'RMSE',
        }
    }

    model = G2GCatBoostModel(config=config, random_state=42)
    model.fit(X_train, y_train, feature_names)

    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    # Ensure predictions are within [0, 1]
    assert float(np.min(preds)) >= 0.0
    assert float(np.max(preds)) <= 1.0

