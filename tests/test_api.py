import requests


def test_predict():
    # input that is not a mine
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
    assert response.json()["prediction"] == ["not a mine"]

    # input that is a mine
    test_input_is_mine = {
        "voltage": 0.335347054,
        "height": 0.818181818,
        "soil": 1.0,
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=test_input_is_mine,
        timeout=1,
    )

    print(response.json())

    assert response.status_code == 200
    assert response.json()["prediction"] == ["mine"]


if __name__ == "__main__":
    test_predict()
