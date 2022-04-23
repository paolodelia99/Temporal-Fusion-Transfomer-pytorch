import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class TSDataset(Dataset):
    """
    The TimeSeries Dataset adapter class
    """

    def __init__(self, static_cols, input_cols,
                 input_timesteps, output_timesteps,
                 encoder_len, data: pd.DataFrame, target: pd.DataFrame):
        assert len(data) == len(target)

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.input_size = len(input_cols)
        self.num_static = len(static_cols)
        self.len = len(data)
        self.data = data
        self.target = target
        self.static_cols = static_cols
        self.input_cols = input_cols
        self.encoder_len = encoder_len

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            X_static = Variable(torch.stack(
                [torch.Tensor(self.data[self.static_cols].iloc[i:i + self.input_timesteps].values) for i in
                 range(*idx.indices(len(self.data)))]
            ))
            X = Variable(torch.stack(
                [torch.Tensor(self.data[self.input_cols].iloc[i:i + self.input_timesteps].values) for i in
                 range(*idx.indices(len(self.data)))]
            ))
            y = Variable(torch.stack(
                [torch.Tensor(
                    self.target[['PUN']].iloc[i + self.encoder_len:i + self.encoder_len + self.output_timesteps].values)
                 for i in range(*idx.indices(len(self.data)))]
            ))
            return X_static, X, y
        else:
            end_idx = idx + self.input_timesteps
            start_out_idx = idx + self.encoder_len
            out_end_idx = start_out_idx + self.output_timesteps
            X_static = Variable(torch.Tensor(self.data[self.static_cols].iloc[idx:end_idx].values))
            X = Variable(torch.Tensor(self.data[self.input_cols].iloc[idx:end_idx].values))
            y = Variable(torch.Tensor(self.target[['PUN']].iloc[start_out_idx:out_end_idx].values))

            return X_static, X, y

    def get_dates(self, idx):
        """
        Return the input and the output dates in the initial dataset

        :param idx: index of indeces of the dataset
        :return: the input and the output dates
        """

        if isinstance(idx, slice):
            in_dates = np.stack(
                [self.data.iloc[i:i + self.input_timesteps].index.values for i in range(*idx.indices(len(self.data)))],
                axis=0
            )
            out_dates = np.stack(
                [self.target.iloc[i + self.encoder_len:i + self.encoder_len + self.output_timesteps].index.values for i
                 in range(*idx.indices(len(self.data)))],
                axis=0
            )
            return in_dates, out_dates
        else:
            end_idx = idx + self.input_timesteps
            start_out_idx = idx + self.encoder_len
            out_end_idx = start_out_idx + self.output_timesteps
            in_dates = self.data.iloc[idx:end_idx].index_values
            out_dates = self.target.iloc[start_out_idx:out_end_idx].index_values
            return in_dates, out_dates

    def __len__(self):
        return self.len - self.input_timesteps
