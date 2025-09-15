from contextlib import asynccontextmanager

import dotenv
from fastapi import Depends, FastAPI
from sqlmodel import Session

from api.database import create_db_and_tables, engine
from api.models import Sensor, SensorInput
from api.utils import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    dotenv.load_dotenv()
    yield


app = FastAPI(lifespan=lifespan)
model = load_model()


def get_session():
    with Session(engine) as session:
        yield session


@app.post("/predict", response_model=Sensor)
async def predict(
    *,
    session: Session = Depends(get_session),
    sensor: SensorInput,
) -> Sensor:
    input_data = [[sensor.voltage, sensor.height, sensor.soil]]

    prediction = model.predict(input_data)
    prediction = int(prediction[0]) == 1  # mine = True, no mine = False

    db_sensor = Sensor.model_validate(sensor, update={"mine": prediction})

    session.add(db_sensor)
    session.commit()
    session.refresh(db_sensor)

    return db_sensor
