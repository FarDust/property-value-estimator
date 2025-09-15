import csv
import requests
import pytest

CSV_PATH = "mle-project-challenge-2/data/future_unseen_examples.csv"
API_URL = "http://localhost:8000/predict"

def load_payloads():
    payloads = []
    with open(CSV_PATH, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            payload = {k: float(v) if "." in v or "e" in v else int(v) if v.isdigit() else v for k, v in row.items()}
            payloads.append(payload)
    return payloads

payloads = load_payloads()

@pytest.mark.parametrize("payload", payloads)
def test_api_prediction(payload):
    import os
    import json
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 200, f"Failed for input: {payload}"
    result = response.json()
    assert "predicted_price" in result, f"No predicted_price in response for input: {payload}"
    # Dump the payload and prediction to a JSON file for reporting
    os.makedirs("reports/predictions", exist_ok=True)
    # Use a filename based on zipcode, bedrooms, and bathrooms (overridable)
    key = f"{payload.get('zipcode','unknown')}_{payload.get('bedrooms','x')}_{payload.get('bathrooms','x')}"
    filename = f"reports/predictions/prediction_{key}.json"
    with open(filename, "w") as f:
        json.dump({"input": payload, "result": result}, f, indent=2)
