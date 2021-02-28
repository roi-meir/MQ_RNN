import pathlib

import pandas as pd
import pytorch_lightning as pl

from MQ_RNN import MQ_RNN
from electricity_dataloader import ElectricityDataModule

DATA_DIR = pathlib.Path("data")


def load_data():
    eldata = pd.read_csv(DATA_DIR.joinpath("LD2011_2014.csv"),
                         parse_dates=[0],
                         delimiter=";",
                         decimal=","
                         )
    eldata.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
    eldata = eldata.resample("1H", on="timestamp").mean()

    return eldata


def main():
    eldata = load_data()
    scaled_data = eldata / eldata[eldata != 0].mean() - 1

    # TODO: change number of samples
    dm = ElectricityDataModule(scaled_data, number_of_samples=50)
    model = MQ_RNN(lr=1e-3, hidden_units=64, num_layers=1, input_size=1, context_size=2)
    trainer = pl.Trainer(max_epochs=4, progress_bar_refresh_rate=1, gpus=0)
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()