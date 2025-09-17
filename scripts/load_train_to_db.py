import sqlite3

import pandas as pd


def load_data_to_db(
    db_path: str = "local.db",
    csv_path: str = "data/interim/train.csv",
    table_name: str = "train",
):
    # load from csv
    df = pd.read_csv(csv_path)

    # write in the database
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    load_data_to_db()
