from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from dr_pb.config import ExperimentConfig
from dr_pb.disturbances import DisturbanceSpec, LatentDisturbanceDataset, ReferenceSampler
from dr_pb.losses import RobotsLoss
from dr_pb.plotting import PlotConfig, plot_all_diagnostics, plot_latent_samples
from dr_pb.systems import RobotsSystem, default_mountain_states
from dr_pb.utils import choose_device


# ---------------------------------------------------------------------------
# Plot appearance options. Modify these in VS Code for thesis figures.
# ---------------------------------------------------------------------------
PLOT_OPTIONS = PlotConfig(
    save_pdf=True,
    save_png=False,
    dpi=220,
    background_max_trajectories=120,
    background_alpha=0.16,
    selected_quantiles=("best", 0.25, 0.50, 0.75, 0.90, 0.95, "worst"),
    worst_fraction=0.05,
    show_obstacles=True,
    obstacle_sigma_levels=(1.0, 2.0),
    show_collision_radius_at_closest=True,
    show_target_collision_radius=False,
    show_titles=True,
)
# ---------------------------------------------------------------------------

BASE_POSITION_OFFSET = (4.0, -4.0, -4.0, -4.0)


def load_config(run_dir: Path) -> ExperimentConfig:
    raw = json.loads((run_dir / "config.json").read_text())
    cfg = ExperimentConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def make_loss_and_xbar(cfg: ExperimentConfig, device: torch.device):
    xbar, _ = default_mountain_states(device)
    system = RobotsSystem(xbar=xbar, x_init=None, u_init=None, linear_plant=cfg.linear_plant, k=cfg.spring_const).to(device)
    loss = RobotsLoss(
        xbar=system.xbar,
        Q=torch.eye(system.state_dim, device=device),
        alpha_u=cfg.alpha_u,
        alpha_col=cfg.alpha_col if cfg.collision_avoidance else None,
        alpha_obst=cfg.alpha_obst if cfg.obstacle_avoidance else None,
        n_agents=cfg.n_agents,
        min_dist=cfg.min_dist,
    )
    return loss, xbar


def load_eval_metrics(run_dir: Path, scalar_file: str, array_suffix: str = "") -> dict:
    scalar_path = run_dir / scalar_file
    if not scalar_path.exists():
        return {}
    scalars = json.loads(scalar_path.read_text())
    out = {}
    for name, vals in scalars.items():
        arr_name = f"{name}{array_suffix}_eval.npz"
        arr_path = run_dir / "arrays" / arr_name
        if not arr_path.exists():
            arr_path = run_dir / "arrays" / f"{name}_eval.npz"
        if not arr_path.exists():
            continue
        data = np.load(arr_path)
        merged = dict(vals)
        for k in data.files:
            if data[k].dtype != object:
                merged[k] = data[k]
        out[name] = merged
    return out


def make_latent_plot(run_dir: Path, cfg: ExperimentConfig, device: torch.device, plot_cfg: PlotConfig):
    train_spec = DisturbanceSpec(
        n_agents=cfg.n_agents,
        horizon=cfg.horizon,
        std_pos=cfg.std_pos_train,
        std_force=cfg.std_force_train,
        force_scale=cfg.force_scale,
        base_position_offset=BASE_POSITION_OFFSET,
    )
    ref_spec = DisturbanceSpec(
        n_agents=cfg.n_agents,
        horizon=cfg.horizon,
        std_pos=cfg.std_pos_ref,
        std_force=cfg.std_force_ref,
        force_scale=cfg.force_scale,
        base_position_offset=BASE_POSITION_OFFSET,
    )
    test_spec = DisturbanceSpec(
        n_agents=cfg.n_agents,
        horizon=cfg.horizon,
        std_pos=cfg.std_pos_test,
        std_force=cfg.std_force_test,
        force_scale=cfg.force_scale,
        base_position_offset=BASE_POSITION_OFFSET,
    )
    train_z = LatentDisturbanceDataset(cfg.num_train, train_spec, seed=cfg.seed + 10, device=device).as_tensor()
    test_z = LatentDisturbanceDataset(cfg.num_test, test_spec, seed=cfg.seed + 30, device=device).as_tensor()
    ref_sampler = ReferenceSampler(ref_spec, device=device, seed=cfg.seed + 40)
    ref_z = ref_sampler.sample(max(cfg.num_test, 200))
    plot_latent_samples(train_z, test_z, ref_z, run_dir / "plots" / "latent_disturbance_samples.pdf", n_agents=cfg.n_agents, plot_cfg=plot_cfg)


def make_plot_config(args) -> PlotConfig:
    cfg = PLOT_OPTIONS
    if args.png:
        cfg = replace(cfg, save_png=True)
    if args.no_titles:
        cfg = replace(cfg, show_titles=False)
    if args.background_max is not None:
        cfg = replace(cfg, background_max_trajectories=args.background_max)
    if args.worst_fraction is not None:
        cfg = replace(cfg, worst_fraction=args.worst_fraction)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generate all diagnostic plots for a saved DR-PB run.")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--png", action="store_true", help="Also save PNG copies of all figures.")
    parser.add_argument("--no-titles", action="store_true", help="Remove plot titles for compact thesis insertion.")
    parser.add_argument("--background-max", type=int, default=None, help="Maximum gray background trajectories.")
    parser.add_argument("--worst-fraction", type=float, default=None, help="Fraction used in worst-trajectory plots.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir)
    device = choose_device(args.device)
    plot_cfg = make_plot_config(args)
    loss_fn, xbar = make_loss_and_xbar(cfg, device)

    histories = json.loads((run_dir / "histories.json").read_text()) if (run_dir / "histories.json").exists() else {}
    metrics = load_eval_metrics(run_dir, "metrics.json", array_suffix="")
    if metrics:
        make_latent_plot(run_dir, cfg, device, plot_cfg)
        plot_all_diagnostics(run_dir, metrics, histories, xbar, loss_fn=loss_fn, min_dist=cfg.min_dist, plot_cfg=plot_cfg)

    for file in sorted(run_dir.glob("metrics_h*.json")):
        stem = file.stem  # metrics_h300
        prefix = stem.replace("metrics_", "")  # h300
        metrics_h = load_eval_metrics(run_dir, file.name, array_suffix=f"_{prefix}")
        if metrics_h:
            plot_all_diagnostics(
                run_dir,
                metrics_h,
                history_by_name=None,
                xbar=xbar,
                loss_fn=loss_fn,
                min_dist=cfg.min_dist,
                prefix=prefix,
                plot_cfg=plot_cfg,
            )

    print(f"Diagnostic plots written to {run_dir / 'plots'}")


if __name__ == "__main__":
    main()
