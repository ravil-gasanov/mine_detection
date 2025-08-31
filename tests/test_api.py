import requests


def test_predict():
    test_input = {
        "voltage": 0.283987607,
        "height": 0.181818182,
        "soil": 0.2,
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=test_input,
        timeout=1,
    )

    print(response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()
