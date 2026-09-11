"""Fast integration test for the minimal DR-PB codebase.

Run from repository root:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dr_pb.config import ExperimentConfig
from dr_pb.controller import PerfBoostController, ZeroController
from dr_pb.disturbances import DisturbanceSpec, LatentDisturbanceDataset, ReferenceSampler
from dr_pb.dro import LambdaSearchConfig
from dr_pb.eval import evaluate_controller
from dr_pb.losses import RobotsLoss
from dr_pb.systems import RobotsSystem, default_mountain_states
from dr_pb.trainers import train_dro, train_saa
from dr_pb.utils import set_seed


def make_parts(device: torch.device):
    cfg = ExperimentConfig(
        seed=1,
        device="cpu",
        horizon=5,
        num_train=4,
        num_valid=4,
        num_test=5,
        batch_size=2,
        dro_ref_samples=3,
        epochs=0,
        dim_internal=4,
        dim_nl=4,
    )
    set_seed(cfg.seed)
    xbar, _ = default_mountain_states(device)
    system = RobotsSystem(xbar=xbar, x_init=None, u_init=None).to(device)

    def make_controller():
        return PerfBoostController(
            noiseless_forward=system.noiseless_forward,
            input_init=system.x_init,
            output_init=system.u_init,
            dim_internal=cfg.dim_internal,
            dim_nl=cfg.dim_nl,
        ).to(device)

    loss_fn = RobotsLoss(
        xbar=system.xbar,
        Q=torch.eye(system.state_dim, device=device),
        alpha_u=cfg.alpha_u,
        alpha_col=cfg.alpha_col,
        n_agents=cfg.n_agents,
        min_dist=cfg.min_dist,
    )
    spec = DisturbanceSpec(
        n_agents=cfg.n_agents,
        horizon=cfg.horizon,
        std_pos=0.2,
        std_force=0.0,
        base_position_offset=(4.0, -4.0, -4.0, -4.0),
    )
    train_z = LatentDisturbanceDataset(cfg.num_train, spec, seed=11, device=device).as_tensor()
    valid_z = LatentDisturbanceDataset(cfg.num_valid, spec, seed=12, device=device).as_tensor()
    test_z = LatentDisturbanceDataset(cfg.num_test, spec, seed=13, device=device).as_tensor()
    ref_sampler = ReferenceSampler(spec, device=device, seed=14)
    return cfg, system, make_controller, loss_fn, spec, train_z, valid_z, test_z, ref_sampler


def main() -> None:
    device = torch.device("cpu")
    cfg, system, make_controller, loss_fn, spec, train_z, valid_z, test_z, ref_sampler = make_parts(device)

    zero_metrics = evaluate_controller(ZeroController(system.in_dim).to(device), system, loss_fn, test_z, spec)
    assert zero_metrics["num_rollouts"] == cfg.num_test

    ctl_saa = make_controller()
    out_saa = train_saa(
        ctl_saa,
        system,
        loss_fn,
        train_z,
        valid_z,
        spec,
        epochs=0,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        log_every=1,
        verbose=False,
    )
    assert len(out_saa["history"]) == 1

    ctl_dro = make_controller()
    out_dro = train_dro(
        ctl_dro,
        system,
        loss_fn,
        train_z,
        valid_z,
        spec,
        ref_sampler,
        epsilon=cfg.dro_epsilon,
        rho=cfg.dro_rho,
        ref_samples=cfg.dro_ref_samples,
        epochs=0,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        log_every=1,
        verbose=False,
    )
    assert len(out_dro["history"]) == 1
    assert out_dro["lambda_value"] > 0

    ctl_golden = make_controller()
    out_golden = train_dro(
        ctl_golden,
        system,
        loss_fn,
        train_z,
        valid_z,
        spec,
        ref_sampler,
        epsilon=cfg.dro_epsilon,
        rho=cfg.dro_rho,
        lambda_mode="golden",
        lambda_search=LambdaSearchConfig(enabled=True, max_iter=3),
        auto_feasible_rho=True,
        ref_samples=cfg.dro_ref_samples,
        epochs=0,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        log_every=1,
        verbose=False,
    )
    hist = out_golden["history"]
    assert len(hist) == 1
    assert hist[0].get("lambda_search_success", False)
    assert out_golden["lambda_value"] > 0

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
