
import numpy as np

def generate_beta_value(beta_min, beta_max, t_array):
    """
    Function used to generate the beta value for the diffusion process

    β(t) = β¯min + t(β¯max - β¯min) for t ∈ [ 0, 1 ]

    """
    return beta_min + t_array * (beta_max - beta_min)


def compute_mean_value_noise(t_array, whole_beta_values, index_t):
    """
    We have : p0t(At|A0) = N (At; A0e - Int 1/2 R t0 β(s)ds, I - Ie - Int R t0 β(s)ds)

    """
    # first we compute the integral - Int(0-t) 1/2 β(s)ds
    integral_beta = np.trapz(whole_beta_values[:index_t], t_array[:index_t])

    mean = np.exp(- 0.5 * integral_beta)
    variance = 1 - np.exp(- integral_beta)

    return mean, variance


def compute_mean_value_whole_noise(t_array, whole_beta_values):
    """
    Make the computation of the mean value for all the time step
    """
    mean_values = []
    variance_values = []

    for index_t in range(len(t_array)):
        mean, variance = compute_mean_value_noise(t_array, whole_beta_values, index_t)
        mean_values.append(mean)
        variance_values.append(variance)

    return mean_values, variance_values


def add_noise_to_graph(data, mean_beta, variance):
    """
    Function used to add noise to a data

    Args:
        data (torch.tensor): original the data
        mean_beta (float): mean value of the noise
        variance (float): variance value of the noise

    Returns:
        data_noisy (torch.tensor): noisy data
        gradiant (torch.tensor): gradiant of the log p0t(At|A0) for the noisy data
    """
    # we generate the noise matrix
    mean_beta = data * mean_beta

    noise_data = np.random.normal(mean_beta, np.sqrt(variance), size=mean_beta.shape)

    # now we can compute the gradiant of log p0t(At|A0)
    # we have : d/dA0 log p0t(At|A0) = - (At - mean_beta) / variance
    if variance == 0:
        gradiant = np.zeros(data.shape)
    else:
        gradiant = -(noise_data - mean_beta) / variance

    return noise_data, gradiant