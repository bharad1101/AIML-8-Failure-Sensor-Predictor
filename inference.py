
import joblib
import numpy as np
import pandas as pd


def create_rolling_features(
    df,
    feature_columns,
    window=12
):
    result = df.copy()

    result = result.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    for feature in feature_columns:

        grouped = result.groupby("unit")[feature]

        result[f"{feature}_rolling_mean"] = (
            grouped
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        result[f"{feature}_rolling_std"] = (
            grouped
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        result[f"{feature}_diff"] = (
            grouped.diff().fillna(0)
        )

    return result


def load_model(model_path):
    """
    Load the trained failure prediction package.
    """

    return joblib.load(model_path)


def predict_failure_risk(
    new_sensor_data,
    model_package
):
    """
    Predict approaching failure risk.

    Parameters
    ----------
    new_sensor_data : pandas.DataFrame
        C-MAPSS-style telemetry.

    model_package : dict
        Saved model package.

    Returns
    -------
    pandas.DataFrame
        Failure probabilities and predictions.
    """

    model = model_package["model"]
    scaler = model_package["scaler"]
    feature_columns = model_package["feature_columns"]
    model_features = model_package["model_features"]
    rolling_window = model_package["rolling_window"]

    processed = create_rolling_features(
        new_sensor_data,
        model_features,
        rolling_window
    )

    processed = processed.replace(
        [np.inf, -np.inf],
        np.nan
    )

    processed = processed.dropna()

    X_new = processed[feature_columns]

    X_new_scaled = scaler.transform(X_new)

    probabilities = model.predict_proba(
        X_new_scaled
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    results = processed[
        ["unit", "cycle"]
    ].copy()

    results["failure_probability"] = probabilities

    results["failure_prediction"] = predictions

    return results
