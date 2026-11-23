"""Spatial-basis PINN: neural diffusion clock and a trainable physical D.

This is deliberately not advertised as an unconstrained (x,y,t) MLP.
The full spatial eigenbasis hard-enforces the known linear geometry and IC.
The neural clock q(tau) is learned from KPFM data, while q'=alpha is the
diffusion residual. D does not enter the observation forward() pass.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from torch import nn


class ClockPINN(nn.Module):
    def __init__(self, model, width=24):
        super().__init__()
        alpha = model["D_initial_m2_s"]*model["time_scale_s"]/(model["domain_um"]*1e-6)**2
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(alpha))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_gain = nn.Parameter(torch.tensor(math.log(model["gain_initial"])))
        self.offset = nn.Parameter(torch.tensor(model["offset_initial_V"]))

    def clock(self, tau):
        # q(0)=0 is a hard IC; q>0 for tau>0. Monotonicity follows from
        # physics only after convergence and is checked on independent times.
        return tau*self.net(2*tau-1).exp()

    def signal(self, tau, steady, modes, rates):
        q = self.clock(tau)
        potential = steady+(modes*torch.exp(-q*rates[None, :])).sum(1, keepdim=True)
        return self.log_gain.exp()*potential+self.offset

    def residual(self, tau):
        q = self.clock(tau)
        dq = torch.autograd.grad(q.sum(), tau, create_graph=True)[0]
        return dq-self.log_alpha.exp(), dq


def tensors_for_rows(basis, rows, model, sigma=None):
    points = np.array([[float(row["x_um"]), float(row["y_um"])] for row in rows])
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    steady, modes = basis.observation(unique, sigma)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
    return (tensor(np.array([float(row["t_s"]) for row in rows])[:, None]/model["time_scale_s"]),
            tensor(steady[inverse, None]), tensor(modes[inverse]), tensor(basis.rates),
            tensor(np.array([float(row["voltage_V"]) for row in rows])[:, None]))


def fit(basis, rows, model, training, *, seed=None, D_initial=None, sigma=None):
    seed = training["seed"] if seed is None else seed
    torch.set_num_threads(training["threads"])
    torch.manual_seed(seed)
    local_model = dict(model)
    if D_initial is not None:
        local_model["D_initial_m2_s"] = D_initial
    network = ClockPINN(local_model, training["width"]).double()
    # Strictly exclude every spatial-validation site and the complete held time.
    train_rows = [row for row in rows if int(row["train"])]
    tau, steady, modes, rates, target = tensors_for_rows(basis, train_rows, model, sigma)
    noise = torch.tensor([[float(row["noise_sigma_V"])] for row in train_rows], dtype=torch.float64)
    opt = torch.optim.Adam([
        {"params": network.net.parameters(), "lr": training["learning_rate"]},
        {"params": [network.log_alpha], "lr": .015},
        {"params": [network.log_gain, network.offset], "lr": .003}])
    generator = torch.Generator().manual_seed(seed+100)
    # Dimensionless reference scale is a fixed numerical conditioning choice.
    alpha_scale = training.get("alpha_scale", .1)
    history, start = [], time.perf_counter()
    for step in range(training["steps"]):
        opt.zero_grad()
        collocation = torch.rand((96, 1), generator=generator, dtype=torch.float64).requires_grad_(True)
        prediction = network.signal(tau, steady, modes, rates)
        data_loss = ((prediction-target)/noise).square().mean()
        residual, _ = network.residual(collocation)
        physics_loss = (residual/alpha_scale).square().mean()
        loss = data_loss+training["physics_weight"]*physics_loss
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == training["steps"]-1:
            D = network.log_alpha.exp().item()*(model["domain_um"]*1e-6)**2/model["time_scale_s"]
            history.append({"step": step+1, "loss": loss.item(), "data_loss": data_loss.item(),
                            "physics_loss": physics_loss.item(), "D_m2_s": D})
        if step % 500 == 0 or step == training["steps"]-1:
            print(f"  seed={seed} step={step+1:4d} loss={loss.item():.4f} D={D:.5e}", flush=True)
    all_tensors = tensors_for_rows(basis, rows, model, sigma)
    with torch.no_grad():
        predicted = network.signal(*all_tensors[:4]).numpy().ravel()
    check_tau = torch.linspace(0, 1, 501, dtype=torch.float64)[:, None].requires_grad_(True)
    residual, dq = network.residual(check_tau)
    alpha = network.log_alpha.exp().item()
    err = predicted-np.array([float(row["voltage_V"]) for row in rows])
    train_mask = np.array([bool(int(row["train"])) for row in rows])
    time_mask = np.array([bool(int(row["time_holdout"])) and row["role"] != "spatial_validation" for row in rows])
    space_mask = np.array([row["role"] == "spatial_validation" and not int(row["time_holdout"]) for row in rows])
    rmse = lambda mask: float(np.sqrt(np.mean(err[mask]**2))) if mask.any() else None
    relative = float(torch.sqrt(residual.square().mean()).item()/alpha)
    with torch.no_grad():
        modal_laplacian = (-torch.exp(-network.clock(check_tau)*all_tensors[3][None, :])
                           * all_tensors[3][None, :]
                           * torch.as_tensor(basis.coefficients, dtype=torch.float64)[None, :])
        laplacian_energy = modal_laplacian.square().sum(1, keepdim=True)/basis.surface.mass.sum()
        pde_rms = torch.sqrt((residual.square()*laplacian_energy).mean()).item()
    warnings = []
    if relative > .05:
        warnings.append("Physics residual exceeds 5% of alpha; train longer or adjust weights.")
    if dq.min().item() <= 0:
        warnings.append("Nonmonotonic diffusion clock; inverse has not converged.")
    report = {"method": "spatial_basis_neural_clock_PINN", "D_eff_m2_s": alpha*(model["domain_um"]*1e-6)**2/model["time_scale_s"],
              "D_eff_um2_s": alpha*model["domain_um"]**2/model["time_scale_s"],
              "D_initial_m2_s": local_model["D_initial_m2_s"], "seed": seed,
              "gain": network.log_gain.exp().item(), "offset_V": network.offset.item(),
              "train_RMSE_V": rmse(train_mask), "time_holdout_RMSE_V": rmse(time_mask),
              "spatial_holdout_RMSE_V": rmse(space_mask), "all_holdout_RMSE_V": rmse(~train_mask),
              "physics_clock_relative_RMS": relative, "minimum_clock_derivative": dq.min().item(),
              "interior_PDE_RMS_V_per_tau": pde_rms,
              "interior_PDE_RMS_V_per_s": pde_rms/model["time_scale_s"],
              "n_training": int(train_mask.sum()), "n_heldout": int((~train_mask).sum()),
              "seconds": time.perf_counter()-start, "psf_sigma_um_used": model["psf_sigma_um"] if sigma is None else sigma,
              "warnings": warnings}
    return network, report, history, predicted
