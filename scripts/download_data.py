import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_data():
    # fetch dataset
    land_mines = fetch_ucirepo(id=763)

    # data (as pandas dataframes)
    X = land_mines.data.features
    y = land_mines.data.targets

    # save data as a csv file
    data = pd.concat([X, y], axis=1)
    data.to_csv("data/raw/land_mines.csv", index=False)

    # metadata
    with open("data/raw/land_mines_metadata.txt", "w") as f:
        f.write(str(land_mines.metadata))


if __name__ == "__main__":
    download_data()
