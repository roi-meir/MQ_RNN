import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number-of-gpus', default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    eldata = load_data()
    scaled_data = eldata / eldata[eldata != 0].mean() - 1

    # TODO: change number of samples
    dm = ElectricityDataModule(scaled_data, number_of_samples=50)
    model = MQ_RNN(lr=1e-3, hidden_units=64, num_layers=1, input_size=4, context_size=4)
    trainer = pl.Trainer(max_epochs=2, progress_bar_refresh_rate=1, gpus=args.number_of_gpus)
    trainer.fit(model, dm)
    import IPython;IPython.embed()



if __name__ == '__main__':
    main()