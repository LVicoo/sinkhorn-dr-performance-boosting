from __future__ import annotations

import copy
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .disturbances import DisturbanceSpec, ReferenceSampler, expand_latent_to_trajectory
from .dro import (
    LambdaSearchConfig,
    SinkhornDualLoss,
    SinkhornDualParams,
    estimate_lambda0,
    search_lambda_for_controller,
)
from .eval import evaluate_controller


def _loader_from_z(z: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(z), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _saa_loss(controller, system, loss_fn, z: torch.Tensor, spec: DisturbanceSpec) -> float:
    w = expand_latent_to_trajectory(z, spec)
    x, _, u = system.rollout(controller, w, train=False)
    return float(loss_fn(x, u).detach().cpu().item())


def train_saa(
    controller,
    system,
    loss_fn,
    train_z: torch.Tensor,
    valid_z: torch.Tensor | None,
    spec: DisturbanceSpec,
    epochs: int = 1000,
    lr: float = 2e-3,
    batch_size: int = 5,
    log_every: int = 50,
    return_best: bool = True,
    grad_clip: float | None = 10.0,
    verbose: bool = True,
) -> Dict:
    opt = torch.optim.Adam(controller.parameters(), lr=lr)
    loader = _loader_from_z(train_z, batch_size=batch_size, shuffle=True)
    history = []
    best_state = copy.deepcopy(controller.state_dict())
    best_valid = float("inf")
    start = time.time()
    log_every = max(1, int(log_every))

    for epoch in range(epochs + 1):
        controller.train()
        batch_losses = []
        for (z_batch,) in loader:
            opt.zero_grad(set_to_none=True)
            w_batch = expand_latent_to_trajectory(z_batch, spec)
            x_log, _, u_log = system.rollout(controller, w_batch, train=True)
            loss = loss_fn(x_log, u_log)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        row = {
            "epoch": epoch,
            "train_batch_mean": float(np.mean(batch_losses)) if batch_losses else np.nan,
            "elapsed_sec": time.time() - start,
        }
        if epoch % log_every == 0 or epoch == epochs:
            train_loss = _saa_loss(controller, system, loss_fn, train_z, spec)
            valid_loss = _saa_loss(controller, system, loss_fn, valid_z, spec) if valid_z is not None else train_loss
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = copy.deepcopy(controller.state_dict())
            row.update({"train_saa": train_loss, "valid_saa": valid_loss})
            if verbose:
                print(f"[SAA] epoch={epoch:5d} batch={row['train_batch_mean']:.4f} train={train_loss:.4f} valid={valid_loss:.4f}")
        history.append(row)

    if return_best:
        controller.load_state_dict(best_state)
    return {"history": history, "best_valid": best_valid}


def train_dro(
    controller,
    system,
    loss_fn,
    train_z: torch.Tensor,
    valid_z: torch.Tensor | None,
    spec: DisturbanceSpec,
    ref_sampler: ReferenceSampler,
    epsilon: float = 0.1,
    rho: float = 0.1,
    lambda_value: float | None = None,
    lambda_mode: str = "fixed",
    lambda_search: LambdaSearchConfig | None = None,
    auto_feasible_rho: bool = False,
    rho_feasibility_margin: float = 1e-4,
    ref_samples: int = 32,
    epochs: int = 1000,
    lr: float = 2e-3,
    batch_size: int = 5,
    log_every: int = 50,
    return_best: bool = True,
    grad_clip: float | None = 10.0,
    normalize_cost: bool = True,
    verbose: bool = True,
) -> Dict:
    """Train the PB controller with the Sinkhorn-DRO objective.

    lambda_mode="fixed" preserves the previous behavior: lambda is either supplied
    or estimated once, then kept fixed.

    lambda_mode="golden" performs a simple alternating minimization: at each epoch,
    before updating the REN parameters, lambda is optimized by golden-section search
    for the current controller parameters.
    """
    lambda_mode = str(lambda_mode).lower().strip()
    if lambda_mode not in {"fixed", "golden"}:
        raise ValueError("lambda_mode must be either 'fixed' or 'golden'")

    if lambda_value is None:
        lambda_value = estimate_lambda0(
            controller, system, loss_fn, train_z, spec, ref_sampler, num_ref=max(32, ref_samples), normalize_cost=normalize_cost
        )
        if verbose:
            print(f"[DRO] estimated lambda0={lambda_value:.6g}")

    lambda0 = float(lambda_value)
    lambda_search = lambda_search or LambdaSearchConfig(enabled=(lambda_mode == "golden"))
    lambda_search.enabled = bool(lambda_mode == "golden")

    params = SinkhornDualParams(
        epsilon=epsilon,
        rho=rho,
        lambda_value=float(lambda_value),
        normalize_cost=normalize_cost,
        auto_feasible_rho=bool(auto_feasible_rho),
        rho_feasibility_margin=float(rho_feasibility_margin),
    )
    dro_loss = SinkhornDualLoss(params, spec, ref_sampler, ref_samples=ref_samples)
    opt = torch.optim.Adam(controller.parameters(), lr=lr)
    loader = _loader_from_z(train_z, batch_size=batch_size, shuffle=True)
    history = []
    best_state = copy.deepcopy(controller.state_dict())
    best_valid = float("inf")
    start = time.time()
    log_every = max(1, int(log_every))

    for epoch in range(epochs + 1):
        lambda_stats = {}
        if lambda_search.enabled:
            lambda_stats = search_lambda_for_controller(
                controller=controller,
                system=system,
                loss_fn=loss_fn,
                train_z=train_z,
                spec=spec,
                ref_sampler=ref_sampler,
                ref_samples=ref_samples,
                epsilon=epsilon,
                rho=rho,
                previous_lambda=float(params.lambda_value),
                search_cfg=lambda_search,
                normalize_cost=normalize_cost,
                auto_feasible_rho=bool(auto_feasible_rho),
                rho_feasibility_margin=float(rho_feasibility_margin),
            )
            if bool(lambda_stats.get("lambda_search_success", False)):
                params.lambda_value = float(lambda_stats["lambda_search_lambda_used"])
            else:
                # Keep previous lambda if the fixed-MC search was numerically unusable.
                lambda_stats["lambda_search_lambda_used"] = float(params.lambda_value)

        controller.train()
        batch_vals = []
        last_stats = {}
        for (z_batch,) in loader:
            opt.zero_grad(set_to_none=True)
            loss, last_stats = dro_loss(controller, system, loss_fn, z_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
            opt.step()
            batch_vals.append(float(loss.detach().cpu().item()))

        row = {
            "epoch": epoch,
            "train_dro_batch_mean": float(np.mean(batch_vals)) if batch_vals else np.nan,
            "elapsed_sec": time.time() - start,
            "lambda_mode": lambda_mode,
            "auto_feasible_rho": bool(auto_feasible_rho),
            "lambda0": lambda0,
            "lambda": float(params.lambda_value),
            **lambda_stats,
            **last_stats,
        }
        # last_stats contains "lambda" as well; ensure the actual training lambda remains explicit.
        row["lambda"] = float(params.lambda_value)

        if epoch % log_every == 0 or epoch == epochs:
            train_saa = _saa_loss(controller, system, loss_fn, train_z, spec)
            valid_saa = _saa_loss(controller, system, loss_fn, valid_z, spec) if valid_z is not None else train_saa
            if valid_saa < best_valid:
                best_valid = valid_saa
                best_state = copy.deepcopy(controller.state_dict())
            row.update({"train_saa": train_saa, "valid_saa": valid_saa})
            if verbose:
                msg = (
                    f"[DRO] epoch={epoch:5d} dro_batch={row['train_dro_batch_mean']:.4f} "
                    f"train_saa={train_saa:.4f} valid_saa={valid_saa:.4f} lambda={params.lambda_value:.4g}"
                )
                if lambda_search.enabled:
                    msg += (
                        f" search_obj={row.get('lambda_search_objective_used', np.nan):.4g}"
                        f" evals={int(row.get('lambda_search_num_evals', 0))}"
                    )
                print(msg)
        history.append(row)

    if return_best:
        controller.load_state_dict(best_state)
    return {
        "history": history,
        "best_valid": best_valid,
        "lambda_value": float(params.lambda_value),
        "lambda0": lambda0,
        "lambda_mode": lambda_mode,
    }


def train_dro_lambda_grid(
    make_controller_fn,
    system,
    loss_fn,
    train_z: torch.Tensor,
    valid_z: torch.Tensor | None,
    spec: DisturbanceSpec,
    ref_sampler: ReferenceSampler,
    multipliers: Iterable[float],
    epsilon: float,
    rho: float,
    ref_samples: int,
    epochs: int,
    lr: float,
    batch_size: int,
    log_every: int,
    return_best: bool,
    grad_clip: float | None,
    verbose: bool = True,
) -> List[Dict]:
    base_controller = make_controller_fn()
    lambda0 = estimate_lambda0(base_controller, system, loss_fn, train_z, spec, ref_sampler, num_ref=max(32, ref_samples))
    results = []
    for mult in multipliers:
        lam = max(1e-4, lambda0 * float(mult))
        if verbose:
            print(f"\n[DRO-GRID] multiplier={mult:g}, lambda={lam:.6g}")
        ctl = make_controller_fn()
        out = train_dro(
            ctl,
            system,
            loss_fn,
            train_z,
            valid_z,
            spec,
            ref_sampler,
            epsilon=epsilon,
            rho=rho,
            lambda_value=lam,
            lambda_mode="fixed",
            ref_samples=ref_samples,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            log_every=log_every,
            return_best=return_best,
            grad_clip=grad_clip,
            verbose=verbose,
        )
        valid_metrics = evaluate_controller(ctl, system, loss_fn, valid_z if valid_z is not None else train_z, spec)
        results.append({"controller": ctl, "train_output": out, "lambda": lam, "multiplier": mult, "valid_metrics": valid_metrics})
    return results
