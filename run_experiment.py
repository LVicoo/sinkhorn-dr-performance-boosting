from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from dr_pb.config import ExperimentConfig
from dr_pb.controller import PerfBoostController, ZeroController
from dr_pb.disturbances import DisturbanceSpec, LatentDisturbanceDataset, ReferenceSampler
from dr_pb.eval import evaluate_controller, save_eval_arrays
from dr_pb.losses import RobotsLoss
from dr_pb.systems import RobotsSystem, default_mountain_states
from dr_pb.dro import LambdaSearchConfig
from dr_pb.trainers import train_dro, train_saa
from dr_pb.utils import choose_device, save_json, set_seed, timestamped_dir, write_text


BASE_POSITION_OFFSET = (4.0, -4.0, -4.0, -4.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAA and/or Sinkhorn-DRO PB controllers on the mountain scenario.")
    p.add_argument("--mode", choices=["saa", "dro", "both", "grid"], default="both")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--num-train", type=int, default=None)
    p.add_argument("--num-valid", type=int, default=None)
    p.add_argument("--num-test", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--lambda-value", type=float, default=None)
    p.add_argument("--lambda-mode", choices=["fixed", "golden"], default=None)
    p.add_argument("--auto-feasible-rho", dest="auto_feasible_rho", action="store_true")
    p.add_argument("--no-auto-feasible-rho", dest="auto_feasible_rho", action="store_false")
    p.set_defaults(auto_feasible_rho=None)
    p.add_argument("--rho-feasibility-margin", type=float, default=None)
    p.add_argument("--lambda-multipliers", type=float, nargs="*", default=None)
    p.add_argument("--lambda-search-low-factor", type=float, default=None)
    p.add_argument("--lambda-search-high-factor", type=float, default=None)
    p.add_argument("--lambda-search-min", type=float, default=None)
    p.add_argument("--lambda-search-max", type=float, default=None)
    p.add_argument("--lambda-search-max-iter", type=int, default=None)
    p.add_argument("--lambda-search-tol", type=float, default=None)
    p.add_argument("--lambda-search-smoothing", type=float, default=None)
    p.add_argument("--lambda-search-max-train-samples", type=int, default=None)
    p.add_argument("--ref-samples", type=int, default=None)
    p.add_argument("--std-pos-train", type=float, default=None)
    p.add_argument("--std-force-train", type=float, default=None)
    p.add_argument("--std-pos-ref", type=float, default=None)
    p.add_argument("--std-force-ref", type=float, default=None)
    p.add_argument("--std-pos-test", type=float, default=None)
    p.add_argument("--std-force-test", type=float, default=None)
    p.add_argument("--alpha-col", type=float, default=None)
    p.add_argument("--alpha-obst", type=float, default=None)
    p.add_argument("--min-dist", type=float, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--results-root", type=str, default=None)
    p.add_argument("--long-horizon", type=int, default=None, help="Optional extra evaluation horizon for trained controllers.")
    p.add_argument("--obstacle-avoidance", dest="obstacle_avoidance", action="store_true")
    p.add_argument("--no-obstacle-avoidance", dest="obstacle_avoidance", action="store_false")
    p.set_defaults(obstacle_avoidance=None)
    p.add_argument("--collision-avoidance", dest="collision_avoidance", action="store_true")
    p.add_argument("--no-collision-avoidance", dest="collision_avoidance", action="store_false")
    p.set_defaults(collision_avoidance=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    mapping = {
        "epochs": "epochs",
        "horizon": "horizon",
        "num_train": "num_train",
        "num_valid": "num_valid",
        "num_test": "num_test",
        "batch_size": "batch_size",
        "lr": "lr",
        "seed": "seed",
        "device": "device",
        "epsilon": "dro_epsilon",
        "rho": "dro_rho",
        "lambda_value": "dro_lambda",
        "lambda_mode": "dro_lambda_mode",
        "rho_feasibility_margin": "dro_rho_feasibility_margin",
        "lambda_search_low_factor": "lambda_search_low_factor",
        "lambda_search_high_factor": "lambda_search_high_factor",
        "lambda_search_min": "lambda_search_min",
        "lambda_search_max": "lambda_search_max",
        "lambda_search_max_iter": "lambda_search_max_iter",
        "lambda_search_tol": "lambda_search_tol",
        "lambda_search_smoothing": "lambda_search_smoothing",
        "lambda_search_max_train_samples": "lambda_search_max_train_samples",
        "ref_samples": "dro_ref_samples",
        "std_pos_train": "std_pos_train",
        "std_force_train": "std_force_train",
        "std_pos_ref": "std_pos_ref",
        "std_force_ref": "std_force_ref",
        "std_pos_test": "std_pos_test",
        "std_force_test": "std_force_test",
        "alpha_col": "alpha_col",
        "alpha_obst": "alpha_obst",
        "min_dist": "min_dist",
        "log_every": "log_every",
        "experiment_name": "experiment_name",
        "results_root": "results_root",
    }
    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if args.obstacle_avoidance is not None:
        cfg.obstacle_avoidance = args.obstacle_avoidance
    if args.collision_avoidance is not None:
        cfg.collision_avoidance = args.collision_avoidance
    if args.auto_feasible_rho is not None:
        cfg.dro_auto_feasible_rho = args.auto_feasible_rho
    if args.lambda_multipliers is not None and len(args.lambda_multipliers) > 0:
        cfg.lambda_grid_multipliers = args.lambda_multipliers
    return cfg


def make_system_controller_loss(cfg: ExperimentConfig, device: torch.device):
    xbar, x0 = default_mountain_states(device)
    system = RobotsSystem(
        xbar=xbar,
        x_init=None,  # mountain initial condition is injected as w_0=x0-xbar.
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

    Q = torch.eye(system.state_dim, device=device)
    loss = RobotsLoss(
        xbar=system.xbar,
        Q=Q,
        alpha_u=cfg.alpha_u,
        alpha_col=cfg.alpha_col if cfg.collision_avoidance else None,
        alpha_obst=cfg.alpha_obst if cfg.obstacle_avoidance else None,
        n_agents=cfg.n_agents,
        min_dist=cfg.min_dist,
    )
    return system, make_controller, loss, xbar, x0


def make_datasets(cfg: ExperimentConfig, device: torch.device):
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
    train = LatentDisturbanceDataset(cfg.num_train, train_spec, seed=cfg.seed + 10, device=device).as_tensor()
    valid = LatentDisturbanceDataset(cfg.num_valid, test_spec, seed=cfg.seed + 20, device=device).as_tensor()
    test = LatentDisturbanceDataset(cfg.num_test, test_spec, seed=cfg.seed + 30, device=device).as_tensor()
    ref_sampler = ReferenceSampler(ref_spec, device=device, seed=cfg.seed + 40)
    return train_spec, ref_spec, test_spec, train, valid, test, ref_sampler


def make_lambda_search_config(cfg: ExperimentConfig) -> LambdaSearchConfig:
    return LambdaSearchConfig(
        enabled=(cfg.dro_lambda_mode == "golden"),
        low_factor=cfg.lambda_search_low_factor,
        high_factor=cfg.lambda_search_high_factor,
        lambda_min=cfg.lambda_search_min,
        lambda_max=cfg.lambda_search_max,
        max_iter=cfg.lambda_search_max_iter,
        tol=cfg.lambda_search_tol,
        smoothing=cfg.lambda_search_smoothing,
        max_train_samples=cfg.lambda_search_max_train_samples,
    )


def summarize_metrics(metrics):
    return {k: v for k, v in metrics.items() if k not in {"costs", "x_log", "u_log", "z", "final_target_dist_per_rollout", "min_interagent_per_rollout", "obstacle_penalty_raw_per_rollout"}}


def evaluate_and_store(name: str, controller, system, loss_fn, z, spec, out_dir: Path, metrics_by_name: dict, controllers_by_name: dict):
    metrics = evaluate_controller(controller, system, loss_fn, z, spec)
    metrics_by_name[name] = metrics
    controllers_by_name[name] = controller
    save_eval_arrays(out_dir, name, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(ExperimentConfig(), args)
    device = choose_device(cfg.device)
    set_seed(cfg.seed)

    out_dir = timestamped_dir(cfg.results_root, f"{cfg.experiment_name}_{args.mode}")
    save_json(out_dir / "config.json", {**cfg.to_dict(), "actual_device": str(device), "mode": args.mode})

    system, make_controller, loss_fn, xbar, _ = make_system_controller_loss(cfg, device)
    train_spec, ref_spec, test_spec, train_z, valid_z, test_z, ref_sampler = make_datasets(cfg, device)
    # Use the reference spec for DRO normalized transport cost. The expansion itself is unaffected by std values.
    # Evaluation uses test_spec.
    spec = ref_spec

    readme = (
        f"Results for mode={args.mode}. Device={device}.\n"
        f"Train std_pos={cfg.std_pos_train}, std_force={cfg.std_force_train}.\n"
        f"Reference std_pos={cfg.std_pos_ref}, std_force={cfg.std_force_ref}.\n"
        f"Test std_pos={cfg.std_pos_test}, std_force={cfg.std_force_test}.\n"
        f"Collision avoidance={cfg.collision_avoidance}, obstacle avoidance={cfg.obstacle_avoidance}.\n"
    )
    write_text(out_dir / "README_run.txt", readme)

    metrics_by_name = {}
    history_by_name = {}
    controllers_by_name = {}

    # Baselines.
    ctl_zero = ZeroController(system.in_dim).to(device)
    evaluate_and_store("prestable_zero", ctl_zero, system, loss_fn, test_z, test_spec, out_dir, metrics_by_name, controllers_by_name)

    ctl_random = make_controller()
    evaluate_and_store("random_ren", ctl_random, system, loss_fn, test_z, test_spec, out_dir, metrics_by_name, controllers_by_name)

    if args.mode in {"saa", "both"}:
        ctl_saa = make_controller()
        out = train_saa(
            ctl_saa,
            system,
            loss_fn,
            train_z,
            valid_z,
            spec,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            log_every=cfg.log_every,
            return_best=cfg.return_best,
            grad_clip=cfg.grad_clip,
        )
        history_by_name["saa"] = out["history"]
        torch.save(ctl_saa.state_dict(), out_dir / "checkpoints" / "controller_saa.pt")
        evaluate_and_store("saa", ctl_saa, system, loss_fn, test_z, test_spec, out_dir, metrics_by_name, controllers_by_name)

    if args.mode in {"dro", "both"}:
        ctl_dro = make_controller()
        out = train_dro(
            ctl_dro,
            system,
            loss_fn,
            train_z,
            valid_z,
            spec,
            ref_sampler,
            epsilon=cfg.dro_epsilon,
            rho=cfg.dro_rho,
            lambda_value=cfg.dro_lambda,
            lambda_mode=cfg.dro_lambda_mode,
            lambda_search=make_lambda_search_config(cfg),
            auto_feasible_rho=cfg.dro_auto_feasible_rho,
            rho_feasibility_margin=cfg.dro_rho_feasibility_margin,
            ref_samples=cfg.dro_ref_samples,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            log_every=cfg.log_every,
            return_best=cfg.return_best,
            grad_clip=cfg.grad_clip,
            normalize_cost=cfg.latent_cost_on_normalized,
        )
        history_by_name["dro"] = out["history"]
        torch.save({"state_dict": ctl_dro.state_dict(), "lambda": out["lambda_value"]}, out_dir / "checkpoints" / "controller_dro.pt")
        evaluate_and_store("dro", ctl_dro, system, loss_fn, test_z, test_spec, out_dir, metrics_by_name, controllers_by_name)

    if args.mode == "grid":
        multipliers = cfg.lambda_grid_multipliers or [0.1, 0.3, 1.0, 3.0, 10.0]
        from dr_pb.dro import estimate_lambda0

        base = make_controller()
        lambda0 = estimate_lambda0(base, system, loss_fn, train_z, spec, ref_sampler, num_ref=max(32, cfg.dro_ref_samples))
        grid_rows = []
        best_name, best_metric = None, float("inf")
        for mult in multipliers:
            lam = max(1e-4, lambda0 * float(mult))
            name = f"dro_lam_{mult:g}x"
            ctl = make_controller()
            out = train_dro(
                ctl,
                system,
                loss_fn,
                train_z,
                valid_z,
                spec,
                ref_sampler,
                epsilon=cfg.dro_epsilon,
                rho=cfg.dro_rho,
                lambda_value=lam,
                ref_samples=cfg.dro_ref_samples,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                log_every=cfg.log_every,
                return_best=cfg.return_best,
                grad_clip=cfg.grad_clip,
                normalize_cost=cfg.latent_cost_on_normalized,
            )
            history_by_name[name] = out["history"]
            torch.save({"state_dict": ctl.state_dict(), "lambda": lam, "multiplier": mult}, out_dir / "checkpoints" / f"controller_{name}.pt")
            valid_metrics = evaluate_controller(ctl, system, loss_fn, valid_z, test_spec)
            test_metrics = evaluate_controller(ctl, system, loss_fn, test_z, test_spec)
            metrics_by_name[name] = test_metrics
            controllers_by_name[name] = ctl
            save_eval_arrays(out_dir, name, test_metrics)
            row = {"name": name, "multiplier": float(mult), "lambda": lam, "valid_mean": valid_metrics["mean_cost"], "test_mean": test_metrics["mean_cost"]}
            grid_rows.append(row)
            if valid_metrics["mean_cost"] < best_metric:
                best_metric = valid_metrics["mean_cost"]
                best_name = name
        save_json(out_dir / "lambda_grid.json", {"lambda0": lambda0, "rows": grid_rows, "best_by_valid_mean": best_name})

    # Save histories and scalar metrics before plotting.
    save_json(out_dir / "metrics.json", {k: summarize_metrics(v) for k, v in metrics_by_name.items()})
    save_json(out_dir / "histories.json", history_by_name)

    if args.long_horizon is not None and args.long_horizon > cfg.horizon:
        long_spec = DisturbanceSpec(
            n_agents=cfg.n_agents,
            horizon=args.long_horizon,
            std_pos=cfg.std_pos_test,
            std_force=cfg.std_force_test,
            force_scale=cfg.force_scale,
            base_position_offset=test_spec.base_position_offset,
        )
        long_metrics_by_name = {}
        for name, ctl in controllers_by_name.items():
            # Skip random_ren for long thesis diagnostics unless it is explicitly useful.
            if name == "random_ren":
                continue
            metrics = evaluate_controller(ctl, system, loss_fn, test_z, long_spec)
            long_metrics_by_name[name] = metrics
            save_eval_arrays(out_dir, f"{name}_h{args.long_horizon}", metrics)
        save_json(out_dir / f"metrics_h{args.long_horizon}.json", {k: summarize_metrics(v) for k, v in long_metrics_by_name.items()})

    if not args.no_plots:
        import subprocess
        diag_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "scripts" / "make_diagnostics.py"),
            str(out_dir),
            "--device",
            "cpu",
        ]
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(diag_cmd, check=True, cwd=Path(__file__).resolve().parent, env=env)

    print("\nSaved results to:", out_dir, flush=True)
    print(json.dumps({k: summarize_metrics(v) for k, v in metrics_by_name.items()}, indent=2), flush=True)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
