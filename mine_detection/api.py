import pickle

from fastapi import FastAPI
from pydantic import BaseModel

with open("models/production_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class SensorInput(BaseModel):
    voltage: float
    height: float
    soil: float


@app.post("/predict")
def predict(data: SensorInput):
    input_data = [[data.voltage, data.height, data.soil]]

    prediction = model.predict(input_data)
    prediction = "mine" if int(prediction[0]) == 1 else "not a mine"

    return {"prediction": {prediction}}


if __name__ == "__main__":
    pass
