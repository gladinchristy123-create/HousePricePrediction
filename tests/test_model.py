
import os
import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Three representative California housing profiles
housing_profiles = {
    "Lower Income Area": {
        "MedInc": 2.5,
        "HouseAge": 15.0,
        "AveRooms": 4.0,
        "AveBedrms": 1.2,
        "Population": 1500.0,
        "AveOccup": 4.0,
        "Latitude": 36.0,
        "Longitude": -120.0
    },
    "Average Suburban Area": {
        "MedInc": 5.0,
        "HouseAge": 30.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    },
    "Higher Income Coastal Area": {
        "MedInc": 9.0,
        "HouseAge": 40.0,
        "AveRooms": 7.0,
        "AveBedrms": 1.0,
        "Population": 800.0,
        "AveOccup": 2.5,
        "Latitude": 37.5,
        "Longitude": -122.0
    }
}

@pytest.mark.parametrize("model_name", [
    "linear_regression",
    "ridge_regression",
    "random_forest"
])
@pytest.mark.parametrize("profile_name", housing_profiles.keys())
def test_model_prediction(model_name, profile_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")

    assert os.path.exists(model_path), f"{model_name} model not found"

    model = joblib.load(model_path)

    sample_house = pd.DataFrame([housing_profiles[profile_name]])

    prediction = model.predict(sample_house)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])
    assert prediction[0] > 0, (
        f"{model_name} returned an invalid price "
        f"for {profile_name}: {prediction[0]}"
    )
