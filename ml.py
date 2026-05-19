# 2. Import libraries and modules
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "forest_fires.csv"
FULL_FEATURE_COLUMNS = [
    "X",
    "Y",
    "month",
    "day",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
]
WEATHER_FEATURE_COLUMNS = [
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
]


def build_model_specs():
    return {
        "gradient_boosting_regressor.pkl": {
            "estimator": GradientBoostingRegressor(random_state=123),
            "feature_columns": FULL_FEATURE_COLUMNS,
            "params": {
                "gradientboostingregressor__n_estimators": [100, 200],
                "gradientboostingregressor__learning_rate": [0.03, 0.05],
                "gradientboostingregressor__max_depth": [2, 3],
                "gradientboostingregressor__subsample": [0.8, 1.0],
                "gradientboostingregressor__min_samples_split": [2, 5],
                "gradientboostingregressor__min_samples_leaf": [1, 2],
            },
        },
        "random_forest_regressor.pkl": {
            "estimator": RandomForestRegressor(random_state=123, n_jobs=-1),
            "feature_columns": FULL_FEATURE_COLUMNS,
            "params": {
                "randomforestregressor__n_estimators": [200, 400],
                "randomforestregressor__max_depth": [None, 12],
                "randomforestregressor__min_samples_split": [2, 5],
                "randomforestregressor__min_samples_leaf": [1, 2],
            },
        },
        "extra_trees_regressor.pkl": {
            "estimator": ExtraTreesRegressor(random_state=123, n_jobs=-1),
            "feature_columns": FULL_FEATURE_COLUMNS,
            "params": {
                "extratreesregressor__n_estimators": [200, 400],
                "extratreesregressor__max_depth": [None, 12],
                "extratreesregressor__min_samples_split": [2, 5],
                "extratreesregressor__min_samples_leaf": [1, 2],
            },
        },
        "weather_only_random_forest_regressor.pkl": {
            "estimator": RandomForestRegressor(random_state=123, n_jobs=-1),
            "feature_columns": WEATHER_FEATURE_COLUMNS,
            "params": {
                "randomforestregressor__n_estimators": [200, 400],
                "randomforestregressor__max_depth": [None, 12],
                "randomforestregressor__min_samples_split": [2, 5],
                "randomforestregressor__min_samples_leaf": [1, 2],
            },
        },
        "weather_only_extra_trees_regressor.pkl": {
            "estimator": ExtraTreesRegressor(random_state=123, n_jobs=-1),
            "feature_columns": WEATHER_FEATURE_COLUMNS,
            "params": {
                "extratreesregressor__n_estimators": [200, 400],
                "extratreesregressor__max_depth": [None, 12],
                "extratreesregressor__min_samples_split": [2, 5],
                "extratreesregressor__min_samples_leaf": [1, 2],
            },
        },
        "weather_only_gradient_boosting_regressor.pkl": {
            "estimator": GradientBoostingRegressor(random_state=123),
            "feature_columns": WEATHER_FEATURE_COLUMNS,
            "params": {
                "gradientboostingregressor__n_estimators": [100, 200],
                "gradientboostingregressor__learning_rate": [0.03, 0.05],
                "gradientboostingregressor__max_depth": [2, 3],
                "gradientboostingregressor__subsample": [0.8, 1.0],
                "gradientboostingregressor__min_samples_split": [2, 5],
                "gradientboostingregressor__min_samples_leaf": [1, 2],
            },
        },
    }


def train_model(X_train, y_train, X_test, y_test, model_path, estimator, params):
    pipeline = make_pipeline(estimator)
    clf = GridSearchCV(pipeline, params, cv=5)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(model_path)
    print(r2_score(y_test, predictions))
    print(mean_squared_error(y_test, predictions))

    fitted_model = clf.best_estimator_.named_steps[next(iter(clf.best_estimator_.named_steps))]
    if hasattr(fitted_model, "feature_importances_"):
        importances = pd.Series(
            fitted_model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)
        print(importances)

    joblib.dump(clf, BASE_DIR / model_path)


# 3. Load Matosinhos fires data.
data = pd.read_csv(DATA_PATH, sep=',')

# 4. Split data into training and test sets
y = data["area"]
X = data.drop("area", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
)

# 5. Train and save each model for the web app.
for model_path, spec in build_model_specs().items():
    feature_columns = spec["feature_columns"]
    train_model(
        X_train[feature_columns],
        y_train,
        X_test[feature_columns],
        y_test,
        model_path,
        spec["estimator"],
        spec["params"],
    )