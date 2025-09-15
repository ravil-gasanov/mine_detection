from sqlmodel import Field, SQLModel


class SensorBase(SQLModel):
    voltage: float
    height: float
    soil: float


class SensorInput(SensorBase):
    pass


class Sensor(SensorBase, table=True):
    id: int | None = Field(default=None, primary_key=True)

    mine: bool
