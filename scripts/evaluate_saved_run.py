from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dr_pb.config import ExperimentConfig
from dr_pb.controller import PerfBoostController, ZeroController
from dr_pb.disturbances import DisturbanceSpec, LatentDisturbanceDataset
from dr_pb.eval import evaluate_controller, save_eval_arrays
from dr_pb.losses import RobotsLoss
from dr_pb.plotting import plot_all_diagnostics
from dr_pb.systems import RobotsSystem, default_mountain_states
from dr_pb.utils import choose_device, save_json, set_seed


BASE_POSITION_OFFSET = (4.0, -4.0, -4.0, -4.0)


def load_config(run_dir: Path) -> ExperimentConfig:
    raw = json.loads((run_dir / "config.json").read_text())
    cfg = ExperimentConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def make_system_controller_loss(cfg: ExperimentConfig, device: torch.device):
    xbar, _ = default_mountain_states(device)
    system = RobotsSystem(
        xbar=xbar,
        x_init=None,
        u_init=None,
        linear_plant=cfg.linear_plant,
        k=cfg.spring_const,
    ).to(device)

    def make_controller():
        return PerfBoostController(
            noiseless_forward=system.noiseless_forward,
            input_init=system.x_init,
            output_init=system.u_init,
            dim_internal=cfg.dim_internal,
            dim_nl=cfg.dim_nl,
            output_amplification=cfg.output_amplification,
            initialization_std=cfg.ren_init_std,
            pos_def_tol=cfg.pos_def_tol,
            contraction_rate_lb=cfg.contraction_rate_lb,
        ).to(device)

    loss = RobotsLoss(
        xbar=system.xbar,
        Q=torch.eye(system.state_dim, device=device),
        alpha_u=cfg.alpha_u,
        alpha_col=cfg.alpha_col if cfg.collision_avoidance else None,
        alpha_obst=cfg.alpha_obst if cfg.obstacle_avoidance else None,
        n_agents=cfg.n_agents,
        min_dist=cfg.min_dist,
    )
    return system, make_controller, loss, xbar


def summarize(metrics):
    return {k: v for k, v in metrics.items() if k not in {"costs", "x_log", "u_log", "z", "final_target_dist_per_rollout", "min_interagent_per_rollout", "obstacle_penalty_raw_per_rollout"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved SAA/DRO controllers over a longer horizon.")
    parser.add_argument("run_dir", type=str, help="Path to a saved result directory containing config.json and checkpoints/.")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument("--out-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir)
    if args.num_test is not None:
        cfg.num_test = args.num_test
    device = choose_device(args.device)
    set_seed(cfg.seed)

    system, make_controller, loss_fn, xbar = make_system_controller_loss(cfg, device)
    spec = DisturbanceSpec(
        n_agents=cfg.n_agents,
        horizon=args.horizon,
        std_pos=cfg.std_pos_test,
        std_force=cfg.std_force_test,
        force_scale=cfg.force_scale,
        base_position_offset=BASE_POSITION_OFFSET,
    )
    z_test = LatentDisturbanceDataset(cfg.num_test, spec, seed=cfg.seed + 30, device=device).as_tensor()

    controllers = {"prestable_zero": ZeroController(system.in_dim).to(device)}

    saa_path = run_dir / "checkpoints" / "controller_saa.pt"
    if saa_path.exists():
        ctl = make_controller()
        ctl.load_state_dict(torch.load(saa_path, map_location=device))
        controllers["saa"] = ctl

    dro_path = run_dir / "checkpoints" / "controller_dro.pt"
    if dro_path.exists():
        ctl = make_controller()
        checkpoint = torch.load(dro_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        ctl.load_state_dict(state_dict)
        controllers["dro"] = ctl

    # Also evaluate grid checkpoints, if present.
    for ckpt_path in sorted((run_dir / "checkpoints").glob("controller_dro_lam_*x.pt")):
        name = ckpt_path.stem.replace("controller_", "")
        ctl = make_controller()
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        ctl.load_state_dict(state_dict)
        controllers[name] = ctl

    out_dir = run_dir / (args.out_name or f"long_horizon_{args.horizon}")
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    metrics_by_name = {}
    for name, ctl in controllers.items():
        metrics = evaluate_controller(ctl, system, loss_fn, z_test, spec)
        metrics_by_name[name] = metrics
        save_eval_arrays(out_dir, name, metrics)

    save_json(out_dir / "metrics.json", {k: summarize(v) for k, v in metrics_by_name.items()})
    plot_all_diagnostics(out_dir, metrics_by_name, history_by_name=None, xbar=xbar, loss_fn=loss_fn, min_dist=cfg.min_dist)

    print(json.dumps({k: summarize(v) for k, v in metrics_by_name.items()}, indent=2))
    print(f"Saved long-horizon evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
