from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "path": BASE_DIR / "gradient_boosting_regressor.pkl",
        "feature_columns": [
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
        ],
    },
    "random_forest": {
        "label": "Random Forest",
        "path": BASE_DIR / "random_forest_regressor.pkl",
        "feature_columns": [
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
        ],
    },
    "extra_trees": {
        "label": "Extra Trees",
        "path": BASE_DIR / "extra_trees_regressor.pkl",
        "feature_columns": [
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
        ],
    },
    "weather_only_gradient_boosting": {
        "label": "Weather Only Gradient Boosting",
        "path": BASE_DIR / "weather_only_gradient_boosting_regressor.pkl",
        "feature_columns": ["DMC", "DC", "ISI", "temp", "RH", "wind"],
    },
    "weather_only_random_forest": {
        "label": "Weather Only Random Forest",
        "path": BASE_DIR / "weather_only_random_forest_regressor.pkl",
        "feature_columns": ["DMC", "DC", "ISI", "temp", "RH", "wind"],
    },
    "weather_only_extra_trees": {
        "label": "Weather Only Extra Trees",
        "path": BASE_DIR / "weather_only_extra_trees_regressor.pkl",
        "feature_columns": ["DMC", "DC", "ISI", "temp", "RH", "wind"],
    },
}


MONTH_OPTIONS = [
    (0, "January"),
    (1, "February"),
    (2, "March"),
    (3, "April"),
    (4, "May"),
    (5, "June"),
    (6, "July"),
    (7, "August"),
    (8, "September"),
    (9, "October"),
    (10, "November"),
    (11, "December"),
]


DAY_OPTIONS = [
    (0, "Monday"),
    (1, "Tuesday"),
    (2, "Wednesday"),
    (3, "Thursday"),
    (4, "Friday"),
    (5, "Saturday"),
    (6, "Sunday"),
]


DISPLAY_FIELD_SPECS = [
    {
        "name": "X",
        "label": "X coordinate",
        "type": "number",
        "min": 1,
        "max": 9,
        "step": 1,
        "default": 4,
        "help": "Spatial x-axis coordinate.",
    },
    {
        "name": "Y",
        "label": "Y coordinate",
        "type": "number",
        "min": 2,
        "max": 9,
        "step": 1,
        "default": 4,
        "help": "Spatial y-axis coordinate.",
    },
    {
        "name": "temp",
        "label": "Temperature (C)",
        "type": "number",
        "min": -10,
        "max": 50,
        "step": 0.1,
        "default": 17,
        "help": "Most influential feature in the trained model.",
    },
    {
        "name": "DMC",
        "label": "DMC",
        "type": "number",
        "min": 0,
        "max": 400,
        "step": 0.1,
        "default": 35.8,
        "help": "Duff Moisture Code index.",
    },
    {
        "name": "wind",
        "label": "Wind speed",
        "type": "number",
        "min": 0,
        "max": 30,
        "step": 0.1,
        "default": 4.9,
        "help": "Wind speed in km/h.",
    },
    {
        "name": "DC",
        "label": "DC",
        "type": "number",
        "min": 0,
        "max": 900,
        "step": 0.1,
        "default": 80.8,
        "help": "Drought Code index.",
    },
    {
        "name": "RH",
        "label": "Relative humidity (%)",
        "type": "number",
        "min": 0,
        "max": 100,
        "step": 1,
        "default": 27,
        "help": "Relative humidity percentage.",
    },
    {
        "name": "ISI",
        "label": "ISI",
        "type": "number",
        "min": 0,
        "max": 60,
        "step": 0.1,
        "default": 7.8,
        "help": "Initial Spread Index.",
    },
    {
        "name": "day",
        "label": "Day of week",
        "type": "select",
        "options": DAY_OPTIONS,
        "default": 2,
        "help": "Encoded day used by the trained model.",
    },
    {
        "name": "month",
        "label": "Month",
        "type": "select",
        "options": MONTH_OPTIONS,
        "default": 7,
        "help": "Encoded month used by the trained model.",
    },
]


MODEL_FEATURE_COLUMNS = [
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


def load_available_models():
    models = {}

    for model_key, spec in MODEL_FILES.items():
        if spec["path"].exists():
            models[model_key] = joblib.load(spec["path"])

    return models


AVAILABLE_MODELS = load_available_models()
MODEL_OPTIONS = [
    (model_key, spec["label"])
    for model_key, spec in MODEL_FILES.items()
    if model_key in AVAILABLE_MODELS
]
WEATHER_ONLY_MODEL_KEYS = [
    model_key
    for model_key, spec in MODEL_FILES.items()
    if len(spec.get("feature_columns", [])) == 6
]
DEFAULT_MODEL_KEY = MODEL_OPTIONS[0][0] if MODEL_OPTIONS else None

app = Flask(__name__)


def default_form_values():
    return {spec["name"]: spec["default"] for spec in DISPLAY_FIELD_SPECS}


def parse_features(form_data, feature_columns):
    values = {}

    for spec in DISPLAY_FIELD_SPECS:
        raw_value = form_data.get(spec["name"], spec["default"])
        if spec["type"] == "select":
            values[spec["name"]] = int(raw_value)
        else:
            values[spec["name"]] = float(raw_value)

    ordered_values = [values[column] for column in feature_columns]
    frame = pd.DataFrame([ordered_values], columns=feature_columns)
    return values, frame


def format_month(value):
    return dict(MONTH_OPTIONS).get(int(value), str(value))


def format_day(value):
    return dict(DAY_OPTIONS).get(int(value), str(value))


def get_model_label(model_key):
    return MODEL_FILES.get(model_key, {}).get("label", model_key)


def get_model_feature_columns(model_key):
    return MODEL_FILES.get(model_key, {}).get("feature_columns", MODEL_FEATURE_COLUMNS)


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = default_form_values()
    selected_model = DEFAULT_MODEL_KEY
    result = None
    error = None

    if not AVAILABLE_MODELS:
        error = "No trained model files were found. Run ml.py first to generate them."
    elif request.method == "POST":
        try:
            requested_model = request.form.get("model_name", DEFAULT_MODEL_KEY)
            selected_model = requested_model if requested_model in AVAILABLE_MODELS else DEFAULT_MODEL_KEY
            feature_columns = get_model_feature_columns(selected_model)
            form_values, features = parse_features(request.form, feature_columns)
            predicted_value = float(AVAILABLE_MODELS[selected_model].predict(features)[0])
            estimated_area_ha = max(float(np.expm1(predicted_value)), 0.0)
            result = {
                "raw_prediction": round(predicted_value, 4),
                "area_ha": round(estimated_area_ha, 2),
                "month_name": format_month(form_values["month"]),
                "day_name": format_day(form_values["day"]),
                "model_label": get_model_label(selected_model),
                "feature_summary": "Weather features only" if len(feature_columns) == 6 else "Full feature set",
                "show_date_fields": len(feature_columns) != 6,
            }
        except Exception as exc:  # pragma: no cover - surfaced in UI
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        fields=DISPLAY_FIELD_SPECS,
        values=form_values,
        result=result,
        error=error,
        model_options=MODEL_OPTIONS,
        selected_model=selected_model,
        weather_only_model_keys=WEATHER_ONLY_MODEL_KEYS,
    )


if __name__ == "__main__":
    app.run(debug=True)