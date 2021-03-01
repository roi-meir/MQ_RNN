import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class ElectricityDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 number_of_samples: int,
                 history_length: int = 168,
                 forcasting_length: int = 24):
        self._df = df
        self._number_of_samples = number_of_samples
        self._history_length = history_length
        self._forcasting_length = forcasting_length

        self.hist_len = pd.Timedelta(hours=history_length)
        self.fct_len = pd.Timedelta(hours=forcasting_length)
        self.offset = pd.Timedelta(hours=1)

        self.max_ts = df.index.max() - self.hist_len - self.fct_len + self.offset

        self.sample()

    def sample(self):
        self.start_ts = (self._df != 0).idxmax()

        households = []

        for household in self._df.columns:
            household_start = self.start_ts[household]
            household_nsamples = min(self._number_of_samples,
                                     self._df.loc[household_start:self.max_ts].shape[0])

            household_samples = (self._df
                                 .loc[household_start:self.max_ts]
                                 .index
                                 .to_series()
                                 .sample(household_nsamples, replace=False)
                                 .index)
            households.extend([(household, start_ts) for start_ts in household_samples])

        self.samples = pd.DataFrame(households, columns=("household", "start_ts"))

        # TODO: delete or support
        # Adding calendar features
        self._df["yearly_cycle"] = np.sin(2 * np.pi * self._df.index.dayofyear / 366)
        self._df["weekly_cycle"] = np.sin(2 * np.pi * self._df.index.dayofweek / 7)
        self._df["daily_cycle"] = np.sin(2 * np.pi * self._df.index.hour / 24)
        self.calendar_features = ["yearly_cycle", "weekly_cycle", "daily_cycle"]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        household, start_ts = self.samples.iloc[idx]

        hs, he = start_ts, start_ts + self.hist_len - self.offset
        fs, fe = he + self.offset, he + self.fct_len

        hist_data = self._df.loc[hs:, [household] + self.calendar_features].iloc[:self._history_length]
        fct_data = self._df.loc[fs:, [household] + self.calendar_features].iloc[:self._forcasting_length]

        return (torch.Tensor(hist_data.values),
                torch.Tensor(fct_data.values))
