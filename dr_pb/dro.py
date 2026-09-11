from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .disturbances import DisturbanceSpec, ReferenceSampler, expand_latent_to_trajectory, pairwise_transport_cost


@dataclass
class SinkhornDualParams:
    epsilon: float = 0.1
    rho: float = 0.1
    lambda_value: float = 1.0
    normalize_cost: bool = True
    auto_feasible_rho: bool = False
    rho_feasibility_margin: float = 1e-4


@dataclass
class LambdaSearchConfig:
    """Stable one-dimensional minimization settings for the Sinkhorn dual multiplier.

    The search interval is centered around the previous/initial lambda:
        [low_factor * lambda, high_factor * lambda]
    and then clamped to [lambda_min, lambda_max].
    """

    enabled: bool = False
    low_factor: float = 0.05
    high_factor: float = 20.0
    lambda_min: float = 1e-4
    lambda_max: float = 1e4
    max_iter: int = 20
    tol: float = 1e-3
    smoothing: float = 0.0  # 0 -> use searched value directly; 0.5 -> average with previous value.
    max_train_samples: int | None = 256


class SinkhornDualLoss:
    """Monte-Carlo approximation of the Sinkhorn-DRO dual objective.

    Given empirical samples z_hat_i and reference samples z_k ~ nu, computes

        lambda*rho + lambda*epsilon * mean_i log mean_k exp(
            ell_theta(z_k)/(lambda*epsilon) - c(z_hat_i,z_k)/epsilon
        ).

    Rollout loss ell_theta is computed only on the K reference samples.
    """

    def __init__(
        self,
        params: SinkhornDualParams,
        spec: DisturbanceSpec,
        ref_sampler: ReferenceSampler,
        ref_samples: int = 32,
    ) -> None:
        self.params = params
        self.spec = spec
        self.ref_sampler = ref_sampler
        self.ref_samples = ref_samples

    def __call__(self, controller, system, loss_fn, z_hat_batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
        eps = float(self.params.epsilon)
        lam = float(self.params.lambda_value)
        if eps <= 0 or lam <= 0:
            raise ValueError("epsilon and lambda_value must be positive")

        z_ref = self.ref_sampler.sample(self.ref_samples)
        w_ref = expand_latent_to_trajectory(z_ref, self.spec)
        x_ref, _, u_ref = system.rollout(controller, w_ref, train=True)
        ell_ref = loss_fn.per_sample(x_ref, u_ref)  # [K]

        cost = pairwise_transport_cost(
            z_hat_batch,
            z_ref,
            self.spec,
            normalize=self.params.normalize_cost,
        )  # [B,K]

        dual = sinkhorn_dual_from_terms(
            ell_ref=ell_ref,
            cost=cost,
            epsilon=eps,
            rho=float(self.params.rho),
            lambda_value=lam,
            auto_feasible_rho=bool(self.params.auto_feasible_rho),
            rho_feasibility_margin=float(self.params.rho_feasibility_margin),
        )

        with torch.no_grad():
            log_mean_exp = _log_mean_exp_terms(ell_ref, cost, eps, lam)
            rho_eff, rho_min = effective_sinkhorn_radius(
                float(self.params.rho), cost, eps, bool(self.params.auto_feasible_rho), float(self.params.rho_feasibility_margin)
            )
            stats = {
                "ell_ref_mean": float(ell_ref.detach().mean().cpu()),
                "ell_ref_max": float(ell_ref.detach().max().cpu()),
                "cost_mean": float(cost.detach().mean().cpu()),
                "log_mean_exp_mean": float(log_mean_exp.detach().mean().cpu()),
                "lambda": lam,
                "epsilon": eps,
                "rho": float(self.params.rho),
                "rho_effective": float(rho_eff),
                "rho_feasibility_min_mc": float(rho_min),
                "auto_feasible_rho": bool(self.params.auto_feasible_rho),
            }
        return dual, stats


def _log_mean_exp_terms(ell_ref: torch.Tensor, cost: torch.Tensor, epsilon: float, lambda_value: float) -> torch.Tensor:
    """Return log mean exp over reference samples for each empirical center."""
    eps = float(epsilon)
    lam = float(lambda_value)
    scores = ell_ref[None, :] / (lam * eps) - cost / eps
    return torch.logsumexp(scores, dim=1) - math.log(scores.shape[1])


def monte_carlo_feasibility_radius(cost: torch.Tensor, epsilon: float) -> torch.Tensor:
    """MC estimate of the minimum radius for a nonempty Sinkhorn ball.

    For the original dual form, as lambda -> infinity the objective slope is
        rho + epsilon * mean_i log mean_k exp(-c_ik / epsilon).
    If this quantity is negative, the dual is unbounded below. This function
    returns the MC radius floor that prevents that pathology.
    """
    eps = float(epsilon)
    log_z = torch.logsumexp(-cost / eps, dim=1) - math.log(cost.shape[1])
    return torch.clamp(-eps * log_z.mean(), min=0.0)


def effective_sinkhorn_radius(
    rho: float,
    cost: torch.Tensor,
    epsilon: float,
    auto_feasible_rho: bool = False,
    margin: float = 1e-4,
) -> tuple[float, float]:
    rho_min = float(monte_carlo_feasibility_radius(cost, epsilon).detach().cpu().item())
    if auto_feasible_rho:
        return max(float(rho), rho_min + float(margin)), rho_min
    return float(rho), rho_min


def sinkhorn_dual_from_terms(
    ell_ref: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    rho: float,
    lambda_value: float,
    auto_feasible_rho: bool = False,
    rho_feasibility_margin: float = 1e-4,
) -> torch.Tensor:
    """Sinkhorn dual objective for fixed rollout losses and fixed pairwise costs.

    This is used both for training and for lambda search. It is differentiable w.r.t.
    ell_ref during training, but lambda search calls it under torch.no_grad().
    """
    eps = float(epsilon)
    lam = float(lambda_value)
    if eps <= 0 or lam <= 0:
        return torch.tensor(float("inf"), device=ell_ref.device, dtype=ell_ref.dtype)
    rho_eff, _ = effective_sinkhorn_radius(float(rho), cost, eps, auto_feasible_rho, rho_feasibility_margin)
    log_mean_exp = _log_mean_exp_terms(ell_ref, cost, eps, lam)
    return lam * rho_eff + lam * eps * log_mean_exp.mean()


def estimate_lambda0(
    controller,
    system,
    loss_fn,
    train_z: torch.Tensor,
    spec: DisturbanceSpec,
    ref_sampler: ReferenceSampler,
    num_ref: int = 64,
    normalize_cost: bool = True,
) -> float:
    """Scale-based lambda initializer: median(loss)/median(transport cost)."""
    with torch.no_grad():
        z_ref = ref_sampler.sample(num_ref)
        w_ref = expand_latent_to_trajectory(z_ref, spec)
        x_ref, _, u_ref = system.rollout(controller, w_ref, train=False)
        ell = loss_fn.per_sample(x_ref, u_ref)
        cost = pairwise_transport_cost(train_z, z_ref, spec, normalize=normalize_cost)
        med_loss = torch.median(ell).item()
        med_cost = torch.median(cost).item()
    return max(1e-4, med_loss / (med_cost + 1e-8))


@torch.no_grad()
def prepare_lambda_search_terms(
    controller,
    system,
    loss_fn,
    train_z: torch.Tensor,
    spec: DisturbanceSpec,
    ref_sampler: ReferenceSampler,
    ref_samples: int,
    normalize_cost: bool = True,
    max_train_samples: int | None = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed Monte-Carlo terms used by golden search in one epoch.

    For fixed theta, lambda only appears in the scalar dual expression. Therefore
    the expensive rollout losses and transport matrix can be computed once and
    reused across all lambda objective evaluations.
    """
    was_training = controller.training
    controller.eval()
    if max_train_samples is not None and train_z.shape[0] > int(max_train_samples):
        z_centers = train_z[: int(max_train_samples)]
    else:
        z_centers = train_z
    z_ref = ref_sampler.sample(ref_samples)
    w_ref = expand_latent_to_trajectory(z_ref, spec)
    x_ref, _, u_ref = system.rollout(controller, w_ref, train=False)
    ell_ref = loss_fn.per_sample(x_ref, u_ref).detach()
    cost = pairwise_transport_cost(z_centers, z_ref, spec, normalize=normalize_cost).detach()
    if was_training:
        controller.train()
    return ell_ref, cost


def _objective_float(
    ell_ref: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    rho: float,
    lambda_value: float,
    auto_feasible_rho: bool = False,
    rho_feasibility_margin: float = 1e-4,
) -> float:
    with torch.no_grad():
        val = sinkhorn_dual_from_terms(
            ell_ref, cost, epsilon, rho, lambda_value, auto_feasible_rho, rho_feasibility_margin
        )
        out = float(val.detach().cpu().item())
    if not math.isfinite(out):
        return float("inf")
    return out


def golden_search_lambda_from_terms(
    ell_ref: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    rho: float,
    previous_lambda: float,
    cfg: LambdaSearchConfig,
    auto_feasible_rho: bool = False,
    rho_feasibility_margin: float = 1e-4,
) -> dict:
    """Golden-section search for lambda in the fixed-theta Sinkhorn dual objective."""
    t0 = time.time()
    prev = float(previous_lambda)
    if not math.isfinite(prev) or prev <= 0:
        prev = max(float(cfg.lambda_min), 1.0)

    a = max(float(cfg.lambda_min), prev * float(cfg.low_factor))
    b = min(float(cfg.lambda_max), prev * float(cfg.high_factor))
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= a:
        a = max(1e-8, float(cfg.lambda_min))
        b = max(a * 10.0, min(float(cfg.lambda_max), prev * 10.0))

    # Evaluate endpoints too. If the optimum sits at a boundary, return it.
    fa = _objective_float(ell_ref, cost, epsilon, rho, a, auto_feasible_rho, rho_feasibility_margin)
    fb = _objective_float(ell_ref, cost, epsilon, rho, b, auto_feasible_rho, rho_feasibility_margin)

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = _objective_float(ell_ref, cost, epsilon, rho, c, auto_feasible_rho, rho_feasibility_margin)
    fd = _objective_float(ell_ref, cost, epsilon, rho, d, auto_feasible_rho, rho_feasibility_margin)
    evals = 4

    success = math.isfinite(fc) or math.isfinite(fd) or math.isfinite(fa) or math.isfinite(fb)
    iters = 0
    for iters in range(int(cfg.max_iter)):
        if abs(b - a) <= float(cfg.tol) * max(1.0, abs((a + b) * 0.5)):
            break
        if fc <= fd:
            b, fb = d, fd
            d, fd = c, fc
            h = b - a
            c = a + invphi2 * h
            fc = _objective_float(ell_ref, cost, epsilon, rho, c, auto_feasible_rho, rho_feasibility_margin)
        else:
            a, fa = c, fc
            c, fc = d, fd
            h = b - a
            d = a + invphi * h
            fd = _objective_float(ell_ref, cost, epsilon, rho, d, auto_feasible_rho, rho_feasibility_margin)
        evals += 1

    candidates = [(a, fa), (b, fb), (c, fc), (d, fd)]
    finite = [(lam, val) for lam, val in candidates if math.isfinite(val)]
    if finite:
        lambda_star, obj_star = min(finite, key=lambda x: x[1])
    else:
        lambda_star, obj_star = prev, float("inf")
        success = False

    smoothing = min(max(float(cfg.smoothing), 0.0), 0.99)
    lambda_used = smoothing * prev + (1.0 - smoothing) * lambda_star
    lambda_used = min(max(lambda_used, float(cfg.lambda_min)), float(cfg.lambda_max))
    obj_used = _objective_float(ell_ref, cost, epsilon, rho, lambda_used, auto_feasible_rho, rho_feasibility_margin)

    rho_eff, rho_min = effective_sinkhorn_radius(float(rho), cost, float(epsilon), auto_feasible_rho, rho_feasibility_margin)

    return {
        "lambda_search_lambda": float(lambda_star),
        "lambda_search_lambda_used": float(lambda_used),
        "lambda_search_objective": float(obj_star),
        "lambda_search_objective_used": float(obj_used),
        "lambda_search_prev": float(prev),
        "lambda_search_low": float(a),
        "lambda_search_high": float(b),
        "lambda_search_interval_initial_low": max(float(cfg.lambda_min), prev * float(cfg.low_factor)),
        "lambda_search_interval_initial_high": min(float(cfg.lambda_max), prev * float(cfg.high_factor)),
        "lambda_search_num_evals": int(evals),
        "lambda_search_num_iters": int(iters + 1),
        "lambda_search_success": bool(success),
        "lambda_search_time_sec": float(time.time() - t0),
        "lambda_search_rho_effective": float(rho_eff),
        "lambda_search_rho_feasibility_min_mc": float(rho_min),
        "lambda_search_auto_feasible_rho": bool(auto_feasible_rho),
    }


def search_lambda_for_controller(
    controller,
    system,
    loss_fn,
    train_z: torch.Tensor,
    spec: DisturbanceSpec,
    ref_sampler: ReferenceSampler,
    ref_samples: int,
    epsilon: float,
    rho: float,
    previous_lambda: float,
    search_cfg: LambdaSearchConfig,
    normalize_cost: bool = True,
    auto_feasible_rho: bool = False,
    rho_feasibility_margin: float = 1e-4,
) -> dict:
    """Convenience wrapper: compute fixed terms, then run golden-section search."""
    ell_ref, cost = prepare_lambda_search_terms(
        controller=controller,
        system=system,
        loss_fn=loss_fn,
        train_z=train_z,
        spec=spec,
        ref_sampler=ref_sampler,
        ref_samples=ref_samples,
        normalize_cost=normalize_cost,
        max_train_samples=search_cfg.max_train_samples,
    )
    return golden_search_lambda_from_terms(
        ell_ref=ell_ref,
        cost=cost,
        epsilon=epsilon,
        rho=rho,
        previous_lambda=previous_lambda,
        cfg=search_cfg,
        auto_feasible_rho=auto_feasible_rho,
        rho_feasibility_margin=rho_feasibility_margin,
    )
