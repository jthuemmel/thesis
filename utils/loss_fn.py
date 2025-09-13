import torch
import math
import einops

def get_statistics(prediction: torch.Tensor, dim: int = 1, mode: str = 'ensemble', epsilon: float = 1e-9):
    if mode == 'ensemble':
        mu, sigma = prediction.mean(dim = dim), prediction.std(dim = dim) 
    elif mode == 'parametric':
        mu, sigma = prediction.split(1, dim = dim)
        mu, sigma = mu.squeeze(dim = dim), sigma.squeeze(dim = dim)
    else:
        raise NotImplementedError(f'Mode {mode} not implemented')
    return mu, sigma + epsilon

def f_gaussian_ignorance(observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # I = ½ ln(2π) + ln σ + ½ z^2
    sigma = sigma.clamp(min=1e-6)
    z = (observation - mu) / sigma
    log2pi = 0.5 * math.log(2 * math.pi)
    score = log2pi + torch.log(sigma) + 0.5 * z**2
    return score

def f_gaussian_crps(observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    sqrtPi, sqrtTwo = math.sqrt(math.pi), math.sqrt(2) #precompute constants
    sigma = sigma.clamp(min=1e-6)
    z = (observation - mu) / sigma
    phi = torch.exp(-z ** 2 / 2) / (sqrtTwo * sqrtPi) #standard normal pdf
    score = sigma * (z * torch.erf(z / sqrtTwo) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005
    return score

def f_empirical_crps(observation, ensemble, fair: bool = False):
    n_member = ensemble.shape[-1]
    # weighting factor for the CRPS
    coef = -1 / (n_member * (n_member - 1)) if fair else -1 / (n_member**2)
    # mean absolute error
    absolute_error = torch.mean((ensemble - observation[..., None]).abs(), dim=-1)
    # pairwise differences
    x_pred_sort = torch.sort(ensemble, dim=-1).values
    diff = torch.diff(x_pred_sort, dim=-1)
    weight = torch.arange(1, n_member, dtype=torch.float32, device = observation.device) 
    weight = weight * torch.arange(n_member - 1, 0, -1, dtype=torch.float32, device = observation.device)
    ndim = [1] * x_pred_sort.ndim
    ndim[-1] = len(weight) 
    weight = weight.view(ndim)
    # Calculate the weighted sum of pairwise differences
    ens_var = torch.sum(diff * weight, dim=-1) 
    return absolute_error + ens_var * coef

def f_kernel_crps(observation: torch.Tensor, ensemble: torch.Tensor, fair: bool = False):
    n_member = ensemble.shape[-1]
    # weighting factor for the CRPS
    coef = -1 / (n_member * (n_member - 1)) if fair else -1 / (n_member**2)
    # mean absolute error
    absolute_error = torch.mean((ensemble - observation[..., None]).abs(), dim=-1)
    # pairwise differences
    ens_var = torch.zeros(size=ensemble.shape[:-1], device=ensemble.device)
    for i in range(n_member):  # loop version to reduce memory usage
        ens_var += torch.sum(torch.abs(ensemble[..., i, None] - ensemble[..., i + 1 :]), dim=-1)
    return absolute_error + coef * ens_var

def f_almost_fair_kernel_crps(observation: torch.Tensor, ensemble: torch.Tensor, alpha: float = 1.0):
    n_member = ensemble.shape[-1]
    coef = 1.0 / (2.0 * n_member * (n_member - 1))
    epsilon = (1.0 - alpha) / n_member
    assert n_member > 1, "Ensemble size must be greater than 1."

    var = torch.abs(ensemble.unsqueeze(dim=-1) - ensemble.unsqueeze(dim=-2))
    diag = torch.eye(n_member, dtype=torch.bool, device=ensemble.device)

    absolute_error = torch.abs(ensemble - observation[..., None])
    err_r = einops.repeat(
        absolute_error,
        "... -> ... k",
        k=n_member,
    )

    mem_err = err_r * ~diag
    mem_err_transpose = mem_err.transpose(-1, -2)
    
    return coef * torch.sum(mem_err + mem_err_transpose - (1 - epsilon) * var, dim=(-1, -2))

