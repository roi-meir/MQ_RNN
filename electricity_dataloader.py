import numpy as np
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader

from electricity_dataset import ElectricityDataset


class ElectricityDataModule(pl.LightningDataModule):
    def __init__(self,
                 df: pd.DataFrame,
                 training: float = 0.7,
                 validation: float = 0.2,
                 test: float = 0.1,
                 number_of_samples: int = 100,
                 batch_size: int = 64,
                 history_length: int = 168,
                 forcasting_length: int = 24,
                 workers=6):

        super().__init__()

        if not np.isclose(training + validation + test, 1):
            raise ValueError("training + validation + test should br equal 1", training + validation + test)

        self._df = df
        self._workers = workers
        self._training = int(training * df.shape[1])
        self._validation = int(validation * df.shape[1])
        self._test = df.shape[1] - self._validation - self._training

        self._batch_size = batch_size
        self._number_of_samples = number_of_samples
        self._forcasting_length = forcasting_length
        self._history_length = history_length
        self.split()

    def split(self):
        hh_rand = (self._df
                   .columns
                   .to_series()
                   .sample(self._df.shape[1],
                           replace=False))

        self._training_hh = hh_rand.iloc[:self._training].index
        self._validation_hh = hh_rand.iloc[self._training:(self._validation + self._training)].index
        self._test_hh = hh_rand.iloc[-self._test:].index

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df = self._df[self._training_hh]
            val_df = self._df[self._validation_hh]

            self._training_dataset = ElectricityDataset(train_df,
                                                        number_of_samples=self._number_of_samples,
                                                        history_length=self._history_length,
                                                        forcasting_length=self._forcasting_length)

            self._validation_dataset = ElectricityDataset(val_df,
                                                          number_of_samples=self._number_of_samples,
                                                          history_length=self._history_length,
                                                          forcasting_length=self._forcasting_length)

        if stage == "test" or stage is None:
            test_df = self._df[self._test_hh]
            self._test_dataset = ElectricityDataset(test_df,
                                                    number_of_samples=self._number_of_samples,
                                                    history_length=self._history_length,
                                                    forcasting_length=self._forcasting_length)

    def train_dataloader(self):
        return DataLoader(self._training_dataset, batch_size=self._batch_size, num_workers=self._workers)

    def val_dataloader(self):
        return DataLoader(self._validation_dataset, batch_size=self._batch_size, num_workers=self._workers)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._batch_size, num_workers=self._workers)
