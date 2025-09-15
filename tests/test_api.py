from api.app import Sensor, app, get_session
from fastapi.testclient import TestClient
import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool


# boilerplate code to use an in-memory SQLite database for testing
@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# boilerplate to set up the TestClient
@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_predict_api(client: TestClient, session: Session):
    # input data that should be a mine
    test_input = {
        "voltage": 0.335347054,
        "height": 0.818181818,
        "soil": 1.0,
    }

    # make the POST request to the /predict endpoint
    response = client.post(
        "http://localhost:8000/predict",
        json=test_input,
        timeout=1,
    )

    data = response.json()

    # check that the request was successful
    assert response.status_code == 200

    # check that the response contains the expected fields and values
    assert data["mine"] is True
    assert data["voltage"] == test_input["voltage"]
    assert data["height"] == test_input["height"]
    assert data["soil"] == test_input["soil"]

    # check that the sensor data was saved in the database
    sensor = session.get(Sensor, data["id"])
    assert sensor is not None
    assert sensor.mine == data["mine"]
