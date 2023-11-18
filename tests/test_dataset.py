"""
Test module to test dataset
"""

import pytest

from diffusion_rnn.dataset import TSDataset, DiffusionTSDataset

def test_dataset():
    """
    Test the TSDataset
    """
    dataset = TSDataset(nb_backstep=4, nb_forecaststep=4)
    assert len(dataset) == 1000 - 4 - 4
    assert len(dataset[0][0]) == 4
    assert len(dataset[0][1]) == 4
    
def test_diffusion_dataset():
    """
    Test of the diffusion dataset
    """
    dataset = DiffusionTSDataset(nb_backstep=4, nb_forecaststep=4, nb_time_step=100)
    
    assert len(dataset) == (1000 - 4 - 4) * 100
    
    
    print(dataset[0])
    print(dataset[0]["y"].shape)
    exit()
    
    