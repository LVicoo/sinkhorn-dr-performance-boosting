"""Small launcher for thesis-relevant DR-PB experiments.

Usage from VS Code terminal:
    python launch_experiments.py

Edit SELECTED_PRESET below, or set RUN_ALL = True.
The launcher simply calls run_experiment.py with a readable set of command-line options.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Change this string to pick one experiment.
SELECTED_PRESET = "obstacles_position_only"

# Set to True to run every preset in PRESETS sequentially.
RUN_ALL = False

# Shared defaults are intentionally modest. Increase epochs/num_test for final thesis figures.
COMMON = {
    "mode": "both",
    "epochs": 100,
    "horizon": 100,
    "num_train": 30,
    "num_valid": 40,
    "num_test": 300,
    "batch_size": 5,
    "lr": 2e-3,
    "ref_samples": 32,
    "epsilon": 0.1,
    "rho": 1.0,
    "long_horizon": 300,
    "log_every": 10,
    "device": "auto",
    # "fixed" is the stable previous behavior. Use "golden" to minimize the Sinkhorn dual over lambda each epoch.
    "lambda_mode": "fixed",
}

PRESETS = {
    # Closest to the experiment you asked for: obstacle avoidance active,
    # no force disturbance, and initial-position uncertainty in both train and test.
    "obstacles_position_only": {
        **COMMON,
        "experiment_name": "obstacles_position_only",
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.0,
        "std_force_test": 0.0,
        "obstacle_avoidance": True,
        "collision_avoidance": True,
        "alpha_obst": 10.0,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },

    # Scarce-data version of the same obstacle experiment. This is useful for
    # showing where DRO should help relative to SAA.
    "obstacles_position_only_golden": {
        **COMMON,
        "experiment_name": "obstacles_position_only_golden",
        "lambda_mode": "golden",
        "auto_feasible_rho": True,
        "lambda_search_low_factor": 0.05,
        "lambda_search_high_factor": 20.0,
        "lambda_search_max_iter": 20,
        "lambda_search_smoothing": 0.0,
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.0,
        "std_force_test": 0.0,
        "obstacle_avoidance": True,
        "collision_avoidance": True,
        "alpha_obst": 10.0,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },

    "scarce_obstacles_position_only": {
        **COMMON,
        "experiment_name": "scarce_obstacles_position_only",
        "num_train": 5,
        "batch_size": 5,
        "std_pos_train": 0.2,
        "std_pos_ref": 0.45,
        "std_pos_test": 0.45,
        "std_force_train": 0.0,
        "std_force_ref": 0.0,
        "std_force_test": 0.0,
        "obstacle_avoidance": True,
        "collision_avoidance": True,
        "alpha_obst": 10.0,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },

    # The force-generalization setup from the analyzed run: SAA only sees initial
    # position samples; DRO has a reference prior with shared constant force; test also has force.
    "force_generalization": {
        **COMMON,
        "experiment_name": "force_generalization",
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.25,
        "std_force_test": 0.25,
        "obstacle_avoidance": False,
        "collision_avoidance": True,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },

    # Same as force_generalization, but with very few empirical centers.
    "scarce_force_generalization": {
        **COMMON,
        "experiment_name": "scarce_force_generalization",
        "num_train": 5,
        "batch_size": 5,
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.25,
        "std_force_test": 0.25,
        "obstacle_avoidance": False,
        "collision_avoidance": True,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },

    # Useful for selecting lambda before committing to a final long run.
    "force_generalization_golden": {
        **COMMON,
        "experiment_name": "force_generalization_golden",
        "lambda_mode": "golden",
        "auto_feasible_rho": True,
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.25,
        "std_force_test": 0.25,
        "obstacle_avoidance": False,
        "collision_avoidance": True,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },

    "lambda_grid_force": {
        **COMMON,
        "mode": "grid",
        "experiment_name": "lambda_grid_force",
        "epochs": 80,
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.25,
        "std_force_test": 0.25,
        "lambda_multipliers": [0.1, 0.3, 1.0, 3.0, 10.0],
        "obstacle_avoidance": False,
        "collision_avoidance": True,
    },

    # Harder combined stress test. This is useful later, after the simpler cases behave sensibly.
    "obstacles_and_force": {
        **COMMON,
        "experiment_name": "obstacles_and_force",
        "std_pos_train": 0.2,
        "std_pos_ref": 0.4,
        "std_pos_test": 0.4,
        "std_force_train": 0.0,
        "std_force_ref": 0.20,
        "std_force_test": 0.20,
        "obstacle_avoidance": True,
        "collision_avoidance": True,
        "alpha_obst": 10.0,
        "alpha_col": 100.0,
        "min_dist": 1.0,
    },
}


def add_arg(cmd: list[str], key: str, value):
    if value is None:
        return
    name = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        cmd.append(name if value else "--no-" + key.replace("_", "-"))
    elif isinstance(value, (list, tuple)):
        cmd.append(name)
        cmd.extend(str(v) for v in value)
    else:
        cmd.extend([name, str(value)])


def run_preset(name: str):
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
    opts = PRESETS[name]
    root = Path(__file__).resolve().parent
    before = set((root / "results").glob("*")) if (root / "results").exists() else set()

    train_cmd = [sys.executable, "run_experiment.py"]
    for key, value in opts.items():
        add_arg(train_cmd, key, value)
    train_cmd.append("--no-plots")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    print("\n=== Running preset:", name, "===", flush=True)
    print(" ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True, cwd=root, env=env)

    after = set((root / "results").glob("*"))
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if not new_dirs:
        # Fallback: latest directory matching the experiment name.
        pattern = f"*{opts.get('experiment_name', 'mountain_dr_pb')}_{opts.get('mode', 'both')}"
        new_dirs = sorted((root / "results").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = new_dirs[0]

    diag_cmd = [sys.executable, "scripts/make_diagnostics.py", str(run_dir), "--device", "cpu"]
    print("Generating diagnostics:", " ".join(diag_cmd), flush=True)
    subprocess.run(diag_cmd, check=True, cwd=root, env=env)


def main():
    names = list(PRESETS) if RUN_ALL else [SELECTED_PRESET]
    for name in names:
        run_preset(name)


if __name__ == "__main__":
    main()
