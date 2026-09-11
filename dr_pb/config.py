from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class ExperimentConfig:
    # Reproducibility / device
    seed: int = 5
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # System
    n_agents: int = 2
    horizon: int = 100
    linear_plant: bool = False
    spring_const: float = 1.0

    # Data: low-dimensional latent disturbance z = [position offsets, shared force]
    # train = empirical centers used by SAA and by the outer empirical distribution in DRO
    # ref   = reference distribution nu used inside the Sinkhorn log-expectation
    # test  = final evaluation distribution
    num_train: int = 30
    num_valid: int = 20
    num_test: int = 300
    batch_size: int = 5
    std_pos_train: float = 0.2
    std_force_train: float = 0.0
    std_pos_ref: float = 0.4
    std_force_ref: float = 0.25
    std_pos_test: float = 0.4
    std_force_test: float = 0.25
    force_scale: float = 1.0
    latent_cost_on_normalized: bool = True

    # Controller REN
    dim_internal: int = 8
    dim_nl: int = 8
    ren_init_std: float = 0.1
    output_amplification: float = 1.0
    pos_def_tol: float = 1e-3
    contraction_rate_lb: float = 1.0

    # Loss
    alpha_u: float = 0.1 / 400.0
    alpha_col: float = 100.0
    alpha_obst: float = 10.0
    min_dist: float = 1.0
    obstacle_avoidance: bool = False
    collision_avoidance: bool = True

    # Training
    epochs: int = 1000
    lr: float = 2e-3
    log_every: int = 10  # Console + full train/valid eval cadence. Batch training loss is stored every epoch.
    return_best: bool = True
    grad_clip: Optional[float] = 10.0

    # DRO
    dro_ref_samples: int = 32
    dro_epsilon: float = 0.1
    dro_rho: float = 0.1
    dro_lambda: Optional[float] = None
    # "fixed" preserves the previous behavior: if dro_lambda is None, lambda0 is estimated once and then fixed.
    # "golden" alternates: first optimize lambda by golden-section search, then update REN parameters.
    dro_lambda_mode: str = "fixed"
    lambda_grid_multipliers: Optional[List[float]] = None
    # Keeps fixed-lambda runs unchanged by default. For golden lambda search, turn this on
    # if the chosen rho is below the MC feasibility floor and lambda runs to the upper bound.
    dro_auto_feasible_rho: bool = False
    dro_rho_feasibility_margin: float = 1e-4

    # Golden-section search for the Sinkhorn dual multiplier.
    lambda_search_low_factor: float = 0.05
    lambda_search_high_factor: float = 20.0
    lambda_search_min: float = 1e-4
    lambda_search_max: float = 1e4
    lambda_search_max_iter: int = 20
    lambda_search_tol: float = 1e-3
    lambda_search_smoothing: float = 0.0
    lambda_search_max_train_samples: Optional[int] = 256

    # Output
    results_root: str = "results"
    experiment_name: str = "mountain_dr_pb"

    def to_dict(self):
        return asdict(self)
