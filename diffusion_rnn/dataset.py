"""
Module to generate the dataset for the diffusion RNN.
"""

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from diffusion_rnn.diffusion_tools import (
    generate_beta_value,
    compute_mean_value_whole_noise,
    add_noise_to_graph,
)

MAX_BETA = 10.0
MIN_BETA = 0.1


class TSDataset(Dataset):
    """
    Simple dataset that output a simple time series.
    """

    def __init__(self, nb_backstep=4, nb_forecaststep=4):
        super().__init__()
        self.nb_backstep = nb_backstep
        self.nb_forecaststep = nb_forecaststep
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data) - (self.nb_backstep + self.nb_forecaststep)

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.nb_backstep]
        y = self.data[
            idx + self.nb_backstep : idx + self.nb_backstep + self.nb_forecaststep
        ]

        return X, y

    def generate_data(self):
        """
        Generate the data.
        Randon 1 and 0.
        """
        data = torch.randint(0, 2, (1000, 2))
        return data


class DiffusionTSDataset(Dataset):
    """
    Dataset to generate the diffusion time series.
    Basicly the idea is to add noise to the time series (on the forecast step
    element).
    """

    def __init__(self, nb_backstep=4, nb_forecaststep=4, nb_time_step=100):
        super().__init__()
        self.dataset = TSDataset(nb_backstep, nb_forecaststep)
        self.nb_time_step = nb_time_step

        self.t_array = torch.linspace(0, 1, nb_time_step)
        self.beta_values = generate_beta_value(MIN_BETA, MAX_BETA, self.t_array)

        self.mean_values, self.variance_values = compute_mean_value_whole_noise(
            self.t_array, self.beta_values
        )

    def __len__(self):
        return (len(self.dataset)) * self.nb_time_step

    def __getitem__(self, idx):
        data_id = idx // self.nb_time_step
        timestep = idx % self.nb_time_step

        X, y = self.dataset[data_id]

        data_noisy, gradiant = add_noise_to_graph(
            y,
            self.mean_values[timestep],
            self.variance_values[timestep],
        )
        
        t_value = self.t_array[timestep]

        return {
            "X": X,
            "y": y,
            "data_noisy": data_noisy,
            "gradiant": gradiant,
            "t_value": t_value,
        }
