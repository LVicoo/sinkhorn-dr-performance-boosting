from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .disturbances import expand_latent_to_trajectory


def _positions_numpy(x_log: np.ndarray) -> np.ndarray:
    n_agents = x_log.shape[-1] // 4
    return np.stack([x_log[:, :, [4 * a, 4 * a + 1]] for a in range(n_agents)], axis=2)


@torch.no_grad()
def evaluate_controller(controller, system, loss_fn, z: torch.Tensor, spec) -> Dict[str, float | np.ndarray]:
    w = expand_latent_to_trajectory(z, spec)
    x_log, _, u_log = system.rollout(controller, w, train=False)
    costs = loss_fn.per_sample(x_log, u_log).detach().cpu().numpy()
    x_np = x_log.detach().cpu().numpy()
    u_np = u_log.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()

    pos = _positions_numpy(x_np)
    xbar_np = system.xbar.detach().cpu().numpy().reshape(-1)
    target = np.stack([xbar_np[[4 * a, 4 * a + 1]] for a in range(pos.shape[2])], axis=0)
    target_dist = np.linalg.norm(pos - target[None, None, :, :], axis=-1)
    final_target_dist_per_rollout = target_dist[:, -1, :].mean(axis=1)

    if pos.shape[2] >= 2:
        # For two agents this is the actual inter-agent distance. For more agents,
        # use the minimum pairwise distance.
        pairwise = []
        for i in range(pos.shape[2]):
            for j in range(i + 1, pos.shape[2]):
                pairwise.append(np.linalg.norm(pos[:, :, i, :] - pos[:, :, j, :], axis=-1))
        interagent = np.stack(pairwise, axis=-1)
        min_interagent_per_step = interagent.min(axis=-1)
        min_interagent_per_rollout = min_interagent_per_step.min(axis=1)
    else:
        min_interagent_per_step = np.full((x_np.shape[0], x_np.shape[1]), np.nan)
        min_interagent_per_rollout = np.full((x_np.shape[0],), np.nan)

    control_norm = np.linalg.norm(u_np, axis=-1)

    metrics = {
        "mean_cost": float(np.mean(costs)),
        "median_cost": float(np.median(costs)),
        "q90_cost": float(np.quantile(costs, 0.90)),
        "q95_cost": float(np.quantile(costs, 0.95)),
        "max_cost": float(np.max(costs)),
        "collisions": float(loss_fn.count_collisions(x_log)),
        "collision_rate": float(loss_fn.count_collisions(x_log) / max(1, z.shape[0] * spec.horizon)),
        "final_mean_target_distance": float(np.mean(final_target_dist_per_rollout)),
        "final_q90_target_distance": float(np.quantile(final_target_dist_per_rollout, 0.90)),
        "min_interagent_distance": float(np.nanmin(min_interagent_per_rollout)),
        "q05_min_interagent_distance": float(np.nanquantile(min_interagent_per_rollout, 0.05)),
        "mean_control_norm": float(np.mean(control_norm)),
        "max_control_norm": float(np.max(control_norm)),
        "num_rollouts": int(z.shape[0]),
        "costs": costs,
        "x_log": x_np,
        "u_log": u_np,
        "z": z_np,
        "final_target_dist_per_rollout": final_target_dist_per_rollout,
        "min_interagent_per_rollout": min_interagent_per_rollout,
    }
    if getattr(loss_fn, "alpha_obst", None) is not None and loss_fn.alpha_obst > 0:
        obst_vals = loss_fn.obstacle_loss_per_sample(x_log).detach().cpu().numpy()
        metrics.update(
            {
                "mean_obstacle_penalty_raw": float(np.mean(obst_vals)),
                "q95_obstacle_penalty_raw": float(np.quantile(obst_vals, 0.95)),
                "obstacle_hits_2sigma": float(loss_fn.count_obstacle_hits(x_log, sigma_level=2.0)),
                "obstacle_penalty_raw_per_rollout": obst_vals,
            }
        )
    return metrics


def save_eval_arrays(path: str | Path, prefix: str, metrics: Dict) -> None:
    path = Path(path)
    payload = {
        "costs": metrics["costs"],
        "x_log": metrics["x_log"],
        "u_log": metrics["u_log"],
        "z": metrics.get("z"),
        "final_target_dist_per_rollout": metrics.get("final_target_dist_per_rollout"),
        "min_interagent_per_rollout": metrics.get("min_interagent_per_rollout"),
    }
    if "obstacle_penalty_raw_per_rollout" in metrics:
        payload["obstacle_penalty_raw_per_rollout"] = metrics["obstacle_penalty_raw_per_rollout"]
    np.savez_compressed(path / "arrays" / f"{prefix}_eval.npz", **payload)
