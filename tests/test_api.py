import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# Testing root
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "Hello World"}


# Testing health returns healthy
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"Status": "Healthy"}


# Testing the input missing a required field
def test_predict_missing_field_validation():
    payload = {
        "age": 45.0,
        "avg_glucose_level": 100.0,
        "bmi": 25.0,
        "gender": "Male",
        "work_type": "Private",
        "ever_married": "Yes",
        "residence_type": "Urban",
        "smoking_status": "formerly smoked",
    }
    del payload["age"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# Testing the input provides a perfect input
def test_predict_no_missing_field_validation():
    payload = {
        "age": 45.0,
        "avg_glucose_level": 100.0,
        "bmi": 25.0,
        "gender": "Male",
        "work_type": "Private",
        "ever_married": "Yes",
        "residence_type": "Urban",
        "smoking_status": "formerly smoked",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# Testing incorrect field type
def test_predict_invalid_data_types():
    payload = {
        "age": "not-a-number",  # Should cause validation failure
        "avg_glucose_level": 100.0,
        "bmi": 25.0,
        "gender": "Male",
        "work_type": "Private",
        "ever_married": "Yes",
        "residence_type": "Urban",
        "smoking_status": "never smoked",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# Testing invalid business value
def test_predict_invalid_business_type():
    payload = {
        "age": 20,  # Should cause validation failure
        "avg_glucose_level": 30000.0,
        "bmi": 25.0,
        "gender": "Male",
        "work_type": "Private",
        "ever_married": "Yes",
        "residence_type": "Urban",
        "smoking_status": "never smoked",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    error_detail = response.json()["detail"]
    assert any("avg_glucose_level" in err["loc"] for err in error_detail)


# Ensure columns are ordered correctly
@patch("app.main.model")
@patch("app.main.get_version")
def test_predict_column_ordering(mock_get_version, mock_model):
    mock_prediction = MagicMock()
    mock_prediction.item.return_value = 0
    mock_model.predict.return_value = mock_prediction
    mock_get_version.return_value = "v1.0.0"

    payload = {
        "smoking_status": "never smoked",
        "age": 34.0,
        "residence_type": "Urban",
        "bmi": 22.1,
        "ever_married": "No",
        "gender": "Male",
        "avg_glucose_level": 85.2,
        "work_type": "Private",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    called_df = mock_model.predict.call_args[0][0]

    expected_cols = [
        "age",
        "avg_glucose_level",
        "bmi",
        "gender",
        "work_type",
        "ever_married",
        "residence_type",
        "smoking_status",
    ]

    assert list(called_df.columns) == expected_cols


@patch("app.main.model")
@patch("app.main.get_version")
# Testing stroke success
def test_predict_stroke_success(mock_get_version, mock_model):
    mock_prediction = MagicMock()
    mock_prediction.item.return_value = 1
    mock_model.predict.return_value = mock_prediction
    mock_get_version.return_value = "Version 1"

    payload = {
        "age": 80.0,
        "avg_glucose_level": 240.5,
        "bmi": 34.2,
        "gender": "Female",
        "work_type": "Self-employed",
        "ever_married": "Yes",
        "residence_type": "Rural",
        "smoking_status": "smokes",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == ["Predicted class is Stroke", "Version 1"]


@patch("app.main.model")
@patch("app.main.get_version")
# Testing stroke failure
def test_predict_non_stroke_success(mock_get_version, mock_model):
    mock_prediction = MagicMock()
    mock_prediction.item.return_value = 0
    mock_model.predict.return_value = mock_prediction
    mock_get_version.return_value = "Version 1"

    payload = {
        "age": 34.0,
        "avg_glucose_level": 85.2,
        "bmi": 22.1,
        "gender": "Male",
        "work_type": "Private",
        "ever_married": "No",
        "residence_type": "Urban",
        "smoking_status": "never smoked",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == ["Predicted class is Non-stroke", "Version 1"]


# Mock model failing
@pytest.mark.xfail(reason="Model failure currently raises an unhandled 500 error")
@patch("app.main.model")
@patch("app.main.get_version")
def test_predict_model_exception(mock_get_version, mock_model):
    mock_model.predict.side_effect = RuntimeError("Model inference engine crashed")

    payload = {
        "age": 50.0,
        "avg_glucose_level": 110.0,
        "bmi": 27.0,
        "gender": "Female",
        "work_type": "Private",
        "ever_married": "No",
        "residence_type": "Urban",
        "smoking_status": "never smoked",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 500
