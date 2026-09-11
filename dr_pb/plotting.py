from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
import torch


@dataclass
class PlotConfig:
    """Central plotting options for diagnostics and thesis figures."""

    save_pdf: bool = True
    save_png: bool = False
    dpi: int = 220
    figure_width: float = 7.0
    figure_height: float = 4.2
    trajectory_figsize: tuple[float, float] = (6.7, 6.2)

    background_max_trajectories: int = 120
    background_alpha: float = 0.16
    background_linewidth: float = 0.65
    selected_quantiles: tuple = ("best", 0.25, 0.50, 0.75, 0.90, 0.95, "worst")
    worst_fraction: float = 0.05

    show_obstacles: bool = True
    obstacle_sigma_levels: tuple[float, ...] = (1.0, 2.0)
    show_collision_radius_at_closest: bool = True
    show_target_collision_radius: bool = False
    max_legend_entries: int = 12

    cost_hist_bins: int = 35
    show_titles: bool = True
    grid_alpha: float = 0.25


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finite(vals):
    vals = np.asarray(vals, dtype=float)
    return vals[np.isfinite(vals)]


def _savefig(fig, path: str | Path, cfg: PlotConfig | None = None) -> None:
    cfg = cfg or PlotConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    if cfg.save_png:
        fig.savefig(path.with_suffix(".png"), dpi=cfg.dpi, bbox_inches="tight")
    if not cfg.save_pdf and not cfg.save_png:
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def _clean_axis(ax, cfg: PlotConfig | None = None) -> None:
    cfg = cfg or PlotConfig()
    ax.grid(True, alpha=cfg.grid_alpha, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _legend(ax, fontsize: int = 8, loc: str = "best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep_h, keep_l = [], []
    for h, l in zip(handles, labels):
        if l and not l.startswith("_") and l not in seen:
            seen.add(l)
            keep_h.append(h)
            keep_l.append(l)
    if keep_h:
        ax.legend(keep_h, keep_l, fontsize=fontsize, loc=loc, frameon=True, framealpha=0.9)


def _positions_numpy(x_log):
    x_log = _to_numpy(x_log)
    n_agents = x_log.shape[-1] // 4
    return np.stack([x_log[:, :, [4 * a, 4 * a + 1]] for a in range(n_agents)], axis=2)


def _target_positions(xbar):
    xbar = _to_numpy(xbar).reshape(-1)
    n_agents = len(xbar) // 4
    return np.stack([xbar[[4 * a, 4 * a + 1]] for a in range(n_agents)], axis=0)


def _active_obstacles(loss_fn) -> bool:
    return loss_fn is not None and getattr(loss_fn, "alpha_obst", None) is not None and loss_fn.alpha_obst > 0


def _draw_obstacles(ax, loss_fn, cfg: PlotConfig) -> None:
    if not (cfg.show_obstacles and _active_obstacles(loss_fn)):
        return
    for idx, (center, cov) in enumerate(zip(loss_fn.obstacle_centers, loss_fn.obstacle_covs)):
        c = _to_numpy(center).reshape(2)
        cv = _to_numpy(cov).reshape(2)
        ax.scatter(c[0], c[1], marker="s", s=35, color="black", zorder=6, label="obstacle center" if idx == 0 else None)
        for level in cfg.obstacle_sigma_levels:
            width = 2.0 * level * math.sqrt(float(cv[0]))
            height = 2.0 * level * math.sqrt(float(cv[1]))
            ax.add_patch(
                Ellipse(
                    xy=(c[0], c[1]),
                    width=width,
                    height=height,
                    fill=False,
                    linestyle="--" if level == 1.0 else ":",
                    linewidth=1.0,
                    edgecolor="black",
                    alpha=0.75,
                    label=f"obstacle {level:g} sigma" if idx == 0 else None,
                    zorder=5,
                )
            )


def _draw_target_collision_radius(ax, xbar, min_dist: float | None, cfg: PlotConfig) -> None:
    if not cfg.show_target_collision_radius or min_dist is None or min_dist <= 0:
        return
    target = _target_positions(xbar)
    for idx, p in enumerate(target):
        ax.add_patch(
            Circle(
                xy=(p[0], p[1]),
                radius=0.5 * min_dist,
                fill=False,
                linestyle=":",
                linewidth=1.0,
                edgecolor="black",
                alpha=0.55,
                label="agent radius" if idx == 0 else None,
            )
        )


def _selected_indices_by_cost(costs: np.ndarray, selected_quantiles: Sequence) -> list[int]:
    costs = np.asarray(costs, dtype=float)
    if costs.size == 0:
        return []
    order = np.argsort(costs)
    out: list[int] = []
    for q in selected_quantiles:
        if isinstance(q, str):
            ql = q.lower()
            if ql == "best":
                idx = int(order[0])
            elif ql == "worst":
                idx = int(order[-1])
            elif ql == "median":
                idx = int(order[len(order) // 2])
            else:
                continue
        else:
            qq = float(q)
            rank = int(round(np.clip(qq, 0.0, 1.0) * (len(order) - 1)))
            idx = int(order[rank])
        if idx not in out:
            out.append(idx)
    return out


def _selected_labels(costs: np.ndarray, indices: Sequence[int], selected_quantiles: Sequence) -> dict[int, str]:
    # Keep labels short and deterministic. Duplicates can happen when n is small.
    labels = {}
    used = set()
    for idx, q in zip(indices, selected_quantiles):
        if isinstance(q, str):
            label = q.lower()
        else:
            label = f"q{int(round(100 * float(q)))}"
        if idx not in used:
            labels[idx] = label
            used.add(idx)
    if indices:
        labels.setdefault(indices[0], "best")
        labels.setdefault(indices[-1], "worst")
    return labels


def _draw_closest_collision_circles(ax, pos: np.ndarray, idx: int, color, min_dist: float | None, n_agents: int) -> None:
    if min_dist is None or min_dist <= 0 or n_agents < 2:
        return
    best_t, best_pair, best_d = 0, (0, 1), float("inf")
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            d = np.linalg.norm(pos[idx, :, i, :] - pos[idx, :, j, :], axis=-1)
            t = int(np.argmin(d))
            if float(d[t]) < best_d:
                best_t, best_pair, best_d = t, (i, j), float(d[t])
    for agent in best_pair:
        p = pos[idx, best_t, agent]
        ax.add_patch(
            Circle(
                xy=(p[0], p[1]),
                radius=0.5 * min_dist,
                fill=False,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.75,
                zorder=4,
            )
        )


def _line_style_for_agent(agent_idx: int) -> str:
    return "-" if agent_idx == 0 else "--"


def plot_loss_history(history, path: str | Path, title: str, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    if not history:
        return
    epochs = np.array([h["epoch"] for h in history], dtype=float)
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))

    for key, label, style in [
        ("train_batch_mean", "train batch mean", "-"),
        ("train_dro_batch_mean", "train DRO batch", "-"),
        ("train_saa", "full train SAA", "o-"),
        ("valid_saa", "valid SAA", "s-"),
    ]:
        if any(key in h for h in history):
            xs = [h["epoch"] for h in history if key in h]
            ys = [h.get(key, np.nan) for h in history if key in h]
            ax.plot(xs, ys, style, markersize=3, linewidth=1.1, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("cost")
    if cfg.show_titles:
        ax.set_title(title)
    vals = _finite([v for h in history for k, v in h.items() if k.endswith("mean") or k.endswith("saa")])
    if vals.size and np.nanmax(vals) / max(np.nanmin(vals), 1e-12) > 10:
        ax.set_yscale("log")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_training_comparison(history_by_name: Dict[str, list], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    if not history_by_name:
        return
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    plotted = False
    for name, hist in history_by_name.items():
        if not hist:
            continue
        key = "train_batch_mean" if any("train_batch_mean" in h for h in hist) else "train_dro_batch_mean"
        if any(key in h for h in hist):
            xs = [h["epoch"] for h in hist if key in h]
            ys = [h.get(key, np.nan) for h in hist if key in h]
            ax.plot(xs, ys, label=name, linewidth=1.2)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("batch training objective")
    if cfg.show_titles:
        ax.set_title("Training objective comparison")
    vals = _finite([h.get("train_batch_mean", h.get("train_dro_batch_mean", np.nan)) for hist in history_by_name.values() for h in hist])
    if vals.size and np.nanmax(vals) / max(np.nanmin(vals), 1e-12) > 10:
        ax.set_yscale("log")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_lambda_history(history_by_name: Dict[str, list], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    plotted = False
    for name, hist in history_by_name.items():
        if not hist or not any("lambda" in h for h in hist):
            continue
        xs = [h["epoch"] for h in hist if "lambda" in h]
        ys = [h.get("lambda", np.nan) for h in hist if "lambda" in h]
        ax.plot(xs, ys, label=f"{name} used", linewidth=1.2)
        plotted = True
        if any("lambda_search_lambda" in h for h in hist):
            xs2 = [h["epoch"] for h in hist if "lambda_search_lambda" in h]
            ys2 = [h.get("lambda_search_lambda", np.nan) for h in hist if "lambda_search_lambda" in h]
            ax.plot(xs2, ys2, linestyle="--", linewidth=1.0, label=f"{name} searched")
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("lambda")
    ax.set_yscale("log")
    if cfg.show_titles:
        ax.set_title("Sinkhorn dual multiplier")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_lambda_search_objective(history_by_name: Dict[str, list], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    plotted = False
    for name, hist in history_by_name.items():
        if not hist or not any("lambda_search_objective_used" in h for h in hist):
            continue
        xs = [h["epoch"] for h in hist if "lambda_search_objective_used" in h]
        ys = [h.get("lambda_search_objective_used", np.nan) for h in hist if "lambda_search_objective_used" in h]
        ax.plot(xs, ys, label=name, linewidth=1.2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("searched dual objective")
    if cfg.show_titles:
        ax.set_title("Lambda-search objective before theta update")
    vals = _finite(ax.lines[0].get_ydata()) if ax.lines else np.array([])
    if vals.size and np.all(vals > 0) and np.nanmax(vals) / max(np.nanmin(vals), 1e-12) > 10:
        ax.set_yscale("log")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_trajectories(
    x_log,
    xbar,
    path: str | Path,
    title: str = "closed-loop trajectories",
    max_rollouts: int | None = None,
    loss_fn=None,
    min_dist: float | None = None,
    costs=None,
    selected_indices: Sequence[int] | None = None,
    plot_cfg: PlotConfig | None = None,
) -> None:
    cfg = plot_cfg or PlotConfig()
    x_log = _to_numpy(x_log)
    pos = _positions_numpy(x_log)
    xbar_np = _to_numpy(xbar).reshape(-1)
    target = _target_positions(xbar_np)
    n_rollouts, _, n_agents, _ = pos.shape
    costs_np = np.asarray(costs if costs is not None else np.arange(n_rollouts), dtype=float)

    if selected_indices is None:
        selected_indices = _selected_indices_by_cost(costs_np, cfg.selected_quantiles)
    selected_indices = [int(i) for i in selected_indices if 0 <= int(i) < n_rollouts]
    labels = _selected_labels(costs_np, selected_indices, cfg.selected_quantiles)

    bg_max = cfg.background_max_trajectories if max_rollouts is None else int(max_rollouts)
    bg_indices = np.linspace(0, n_rollouts - 1, min(bg_max, n_rollouts), dtype=int) if n_rollouts else []

    fig, ax = plt.subplots(figsize=cfg.trajectory_figsize)

    for idx in bg_indices:
        for a in range(n_agents):
            ax.plot(
                pos[idx, :, a, 0],
                pos[idx, :, a, 1],
                linestyle=_line_style_for_agent(a),
                color="0.72",
                linewidth=cfg.background_linewidth,
                alpha=cfg.background_alpha,
                zorder=1,
            )

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(10, len(selected_indices))))
    for k, idx in enumerate(selected_indices):
        color = colors[k % len(colors)]
        label = labels.get(idx, f"#{idx}")
        for a in range(n_agents):
            ax.plot(
                pos[idx, :, a, 0],
                pos[idx, :, a, 1],
                linestyle=_line_style_for_agent(a),
                color=color,
                linewidth=1.9,
                alpha=0.95,
                zorder=3,
                label=f"{label}" if a == 0 else None,
            )
            ax.scatter(pos[idx, 0, a, 0], pos[idx, 0, a, 1], s=16, marker="o", color=color, zorder=4)
            ax.scatter(pos[idx, -1, a, 0], pos[idx, -1, a, 1], s=20, marker="x", color=color, zorder=4)
        if cfg.show_collision_radius_at_closest:
            _draw_closest_collision_circles(ax, pos, idx, color, min_dist, n_agents)

    for a in range(n_agents):
        ax.scatter(target[a, 0], target[a, 1], s=90, marker="*", color="black", label="target" if a == 0 else None, zorder=7)

    _draw_obstacles(ax, loss_fn, cfg)
    _draw_target_collision_radius(ax, xbar_np, min_dist, cfg)

    ax.axis("equal")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    if cfg.show_titles:
        ax.set_title(title)
    _clean_axis(ax, cfg)
    _legend(ax, fontsize=7)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_worst_fraction_trajectories(metrics: Dict, xbar, path: str | Path, title: str, loss_fn=None, min_dist=None, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    costs = np.asarray(metrics["costs"], dtype=float)
    if costs.size == 0:
        return
    n_worst = max(1, int(math.ceil(cfg.worst_fraction * len(costs))))
    idx = np.argsort(costs)[-n_worst:]
    # Keep background limited to the same worst set by passing max_rollouts=0 and selected_indices=idx.
    local_cfg = PlotConfig(**{**cfg.__dict__, "background_max_trajectories": 0, "selected_quantiles": tuple(f"w{i+1}" for i in range(len(idx)))})
    plot_trajectories(
        metrics["x_log"],
        xbar,
        path,
        title=title,
        max_rollouts=0,
        loss_fn=loss_fn,
        min_dist=min_dist,
        costs=costs,
        selected_indices=list(idx),
        plot_cfg=local_cfg,
    )


def plot_trajectory_overlay(metrics_by_name: Dict[str, Dict], xbar, path: str | Path, max_rollouts_per_controller: int = 5, loss_fn=None, min_dist=None, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    xbar_np = _to_numpy(xbar).reshape(-1)
    target = _target_positions(xbar_np)
    fig, ax = plt.subplots(figsize=cfg.trajectory_figsize)
    for name, metrics in metrics_by_name.items():
        x_log = _to_numpy(metrics["x_log"])
        pos = _positions_numpy(x_log)
        costs = np.asarray(metrics.get("costs", np.arange(pos.shape[0])))
        selected = _selected_indices_by_cost(costs, ("median", 0.95, "worst"))[:max_rollouts_per_controller]
        for idx in selected:
            for a in range(pos.shape[2]):
                ax.plot(
                    pos[idx, :, a, 0],
                    pos[idx, :, a, 1],
                    linestyle=_line_style_for_agent(a),
                    linewidth=1.0,
                    alpha=0.60,
                    label=name if idx == selected[0] and a == 0 else None,
                )
    for a in range(target.shape[0]):
        ax.scatter(target[a, 0], target[a, 1], s=90, marker="*", color="black", label="target" if a == 0 else None, zorder=7)
    _draw_obstacles(ax, loss_fn, cfg)
    _draw_target_collision_radius(ax, xbar_np, min_dist, cfg)
    ax.axis("equal")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    if cfg.show_titles:
        ax.set_title("Representative trajectory overlay")
    _clean_axis(ax, cfg)
    _legend(ax, fontsize=7)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_cost_boxplot(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    names = list(metrics_by_name.keys())
    data = [np.asarray(metrics_by_name[n]["costs"], dtype=float) for n in names]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.15 * len(names)), cfg.figure_height))
    ax.boxplot(data, labels=names, showfliers=False)
    ax.set_ylabel("per-rollout cost")
    if cfg.show_titles:
        ax.set_title("Test cost distribution")
    ax.tick_params(axis="x", labelrotation=25)
    _clean_axis(ax, cfg)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_cost_histograms(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    for name, metrics in metrics_by_name.items():
        ax.hist(np.asarray(metrics["costs"], dtype=float), bins=cfg.cost_hist_bins, alpha=0.35, density=True, label=name)
    ax.set_xlabel("per-rollout cost")
    ax.set_ylabel("density")
    if cfg.show_titles:
        ax.set_title("Test cost histogram")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_cost_ecdf(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    for name, metrics in metrics_by_name.items():
        vals = np.sort(np.asarray(metrics["costs"], dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=name, linewidth=1.4)
    ax.set_xlabel("per-rollout cost")
    ax.set_ylabel("empirical CDF")
    if cfg.show_titles:
        ax.set_title("Test cost empirical CDF")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_target_distance(metrics_by_name: Dict[str, Dict], xbar, path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    target = _target_positions(xbar)
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    for name, metrics in metrics_by_name.items():
        pos = _positions_numpy(metrics["x_log"])
        dist = np.linalg.norm(pos - target[None, None, :, :], axis=-1)
        ax.plot(np.arange(dist.shape[1]), dist.mean(axis=(0, 2)), label=name, linewidth=1.3)
    ax.set_xlabel("time step")
    ax.set_ylabel("mean distance to target")
    if cfg.show_titles:
        ax.set_title("Mean distance to target")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_final_target_distance_boxplot(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    names = list(metrics_by_name.keys())
    data = [metrics_by_name[n].get("final_target_dist_per_rollout") for n in names]
    if any(d is None for d in data):
        return
    fig, ax = plt.subplots(figsize=(max(5.5, 1.15 * len(names)), cfg.figure_height))
    ax.boxplot(data, labels=names, showfliers=False)
    ax.set_ylabel("final mean target distance")
    if cfg.show_titles:
        ax.set_title("Final target distance distribution")
    ax.tick_params(axis="x", labelrotation=25)
    _clean_axis(ax, cfg)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def _min_interagent_over_time(x_log) -> np.ndarray:
    pos = _positions_numpy(x_log)
    if pos.shape[2] < 2:
        return np.full((pos.shape[0], pos.shape[1]), np.nan)
    dists = []
    for i in range(pos.shape[2]):
        for j in range(i + 1, pos.shape[2]):
            dists.append(np.linalg.norm(pos[:, :, i, :] - pos[:, :, j, :], axis=-1))
    return np.stack(dists, axis=-1).min(axis=-1)


def plot_interagent_distance(metrics_by_name: Dict[str, Dict], path: str | Path, min_dist: float = 1.0, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    for name, metrics in metrics_by_name.items():
        dist = _min_interagent_over_time(metrics["x_log"])
        if np.all(np.isnan(dist)):
            continue
        ax.plot(np.arange(dist.shape[1]), np.nanmean(dist, axis=0), label=f"{name} mean", linewidth=1.2)
        ax.plot(np.arange(dist.shape[1]), np.nanquantile(dist, 0.10, axis=0), linestyle="--", linewidth=1.0, label=f"{name} q10")
    ax.axhline(min_dist, linestyle=":", color="black", linewidth=1.1, label="collision threshold")
    ax.set_xlabel("time step")
    ax.set_ylabel("minimum inter-agent distance")
    if cfg.show_titles:
        ax.set_title("Inter-agent distance")
    _clean_axis(ax, cfg)
    _legend(ax, fontsize=7)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_min_interagent_boxplot(metrics_by_name: Dict[str, Dict], path: str | Path, min_dist: float = 1.0, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    names = list(metrics_by_name.keys())
    data = [metrics_by_name[n].get("min_interagent_per_rollout") for n in names]
    if any(d is None for d in data):
        return
    fig, ax = plt.subplots(figsize=(max(5.5, 1.15 * len(names)), cfg.figure_height))
    ax.boxplot(data, labels=names, showfliers=False)
    ax.axhline(min_dist, linestyle=":", color="black", linewidth=1.1, label="collision threshold")
    ax.set_ylabel("minimum inter-agent distance")
    if cfg.show_titles:
        ax.set_title("Safety-margin distribution")
    ax.tick_params(axis="x", labelrotation=25)
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_control_norm(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    for name, metrics in metrics_by_name.items():
        u = _to_numpy(metrics["u_log"])
        norm = np.linalg.norm(u, axis=-1)
        ax.plot(np.arange(norm.shape[1]), norm.mean(axis=0), label=f"{name} mean", linewidth=1.2)
        ax.plot(np.arange(norm.shape[1]), np.quantile(norm, 0.90, axis=0), linestyle="--", linewidth=1.0, label=f"{name} q90")
    ax.set_xlabel("time step")
    ax.set_ylabel("control norm")
    if cfg.show_titles:
        ax.set_title("Control effort over time")
    _clean_axis(ax, cfg)
    _legend(ax, fontsize=7)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_metrics_bar(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    names = list(metrics_by_name.keys())
    keys = ["mean_cost", "q95_cost", "max_cost", "final_mean_target_distance", "min_interagent_distance", "mean_control_norm"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    axes = axes.ravel()
    for ax, key in zip(axes, keys):
        vals = [metrics_by_name[n].get(key, np.nan) for n in names]
        ax.bar(names, vals)
        ax.set_title(key.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        _clean_axis(ax, cfg)
    if cfg.show_titles:
        fig.suptitle("Scalar performance summary")
    fig.tight_layout()
    _savefig(fig, path, cfg)


def _loss_components(metrics: Dict, loss_fn) -> dict[str, np.ndarray]:
    x = _to_numpy(metrics["x_log"])
    u = _to_numpy(metrics["u_log"])
    xbar = _to_numpy(loss_fn.xbar).reshape(1, 1, -1)
    q = _to_numpy(loss_fn.Q)
    xc = x - xbar
    tracking = np.einsum("...i,ij,...j->...", xc, q, xc)
    control = float(getattr(loss_fn, "alpha_u", 0.0)) * np.sum(u * u, axis=-1)

    out = {"tracking": tracking, "control": control}

    alpha_col = getattr(loss_fn, "alpha_col", None)
    if alpha_col is not None and alpha_col > 0:
        n_agents = int(getattr(loss_fn, "n_agents", x.shape[-1] // 4))
        pos = _positions_numpy(x)
        collision = np.zeros(x.shape[:2], dtype=float)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                d2 = np.sum((pos[:, :, i, :] - pos[:, :, j, :]) ** 2, axis=-1)
                active = d2 < (float(loss_fn.min_dist) + 0.2) ** 2
                collision += (1.0 / (d2 + 1e-3)) * active
        out["collision"] = float(alpha_col) * collision

    alpha_obst = getattr(loss_fn, "alpha_obst", None)
    if alpha_obst is not None and alpha_obst > 0:
        pos = _positions_numpy(x)
        obstacle = np.zeros(x.shape[:2], dtype=float)
        for center, cov in zip(loss_fn.obstacle_centers, loss_fn.obstacle_covs):
            c = _to_numpy(center).reshape(1, 1, 1, 2)
            cv = _to_numpy(cov).reshape(1, 1, 1, 2)
            den = (2 * np.pi) * np.sqrt(np.prod(cv))
            obstacle += np.exp((-0.5 * (pos - c) ** 2 / cv).sum(-1)).sum(axis=-1) / den
        out["obstacle"] = float(alpha_obst) * obstacle

    total = np.zeros_like(tracking)
    for arr in out.values():
        total = total + arr
    out["total"] = total
    return out


def plot_loss_components_bar(metrics_by_name: Dict[str, Dict], loss_fn, path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    if loss_fn is None:
        return
    names = list(metrics_by_name.keys())
    components = ["tracking", "control", "collision", "obstacle"]
    means = {c: [] for c in components}
    present = set()
    for name in names:
        comp = _loss_components(metrics_by_name[name], loss_fn)
        for c in components:
            val = float(np.mean(comp[c])) if c in comp else 0.0
            means[c].append(val)
            if val != 0.0:
                present.add(c)
    components = [c for c in components if c in present]
    if not components:
        return
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.15 * len(names)), cfg.figure_height))
    for c in components:
        vals = np.asarray(means[c])
        ax.bar(x, vals, bottom=bottom, label=c)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("mean weighted component")
    if cfg.show_titles:
        ax.set_title("Mean loss decomposition")
    _clean_axis(ax, cfg)
    _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_loss_components_boxplot(metrics_by_name: Dict[str, Dict], loss_fn, path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    if loss_fn is None:
        return
    labels, data = [], []
    for name, metrics in metrics_by_name.items():
        comp = _loss_components(metrics, loss_fn)
        for c in ["tracking", "control", "collision", "obstacle", "total"]:
            if c in comp:
                labels.append(f"{name}\n{c}")
                data.append(np.mean(comp[c], axis=1))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(data)), 4.8))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("per-rollout mean weighted component")
    if cfg.show_titles:
        ax.set_title("Loss component distributions")
    ax.tick_params(axis="x", labelrotation=65, labelsize=7)
    _clean_axis(ax, cfg)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_loss_components_over_time(metrics_by_name: Dict[str, Dict], loss_fn, out_dir: str | Path, prefix: str = "", plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    if loss_fn is None:
        return
    out_dir = Path(out_dir)
    components_order = ["tracking", "control", "collision", "obstacle", "total"]
    for name, metrics in metrics_by_name.items():
        comp = _loss_components(metrics, loss_fn)
        fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
        for c in components_order:
            if c not in comp:
                continue
            ts = comp[c]
            ax.plot(np.arange(ts.shape[1]), np.mean(ts, axis=0), label=c, linewidth=1.2)
        ax.set_xlabel("time step")
        ax.set_ylabel("mean weighted component")
        if cfg.show_titles:
            ax.set_title(f"Loss components over time: {name}")
        _clean_axis(ax, cfg)
        _legend(ax, fontsize=7)
        fig.tight_layout()
        _savefig(fig, out_dir / f"{prefix}loss_components_over_time_{name}.pdf", cfg)


def plot_obstacle_penalty(metrics_by_name: Dict[str, Dict], path: str | Path, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    names = [n for n, m in metrics_by_name.items() if "obstacle_penalty_raw_per_rollout" in m]
    if not names:
        return
    data = [metrics_by_name[n]["obstacle_penalty_raw_per_rollout"] for n in names]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.15 * len(names)), cfg.figure_height))
    ax.boxplot(data, labels=names, showfliers=False)
    ax.set_ylabel("raw obstacle penalty")
    if cfg.show_titles:
        ax.set_title("Obstacle exposure distribution")
    ax.tick_params(axis="x", labelrotation=25)
    _clean_axis(ax, cfg)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def plot_latent_samples(train_z, test_z, ref_z, path: str | Path, n_agents: int = 2, plot_cfg: PlotConfig | None = None) -> None:
    cfg = plot_cfg or PlotConfig()
    train_z = _to_numpy(train_z)
    test_z = _to_numpy(test_z)
    ref_z = _to_numpy(ref_z) if ref_z is not None else None
    f0 = 2 * n_agents
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    specs = [
        (axes[0], 0, 1, "initial position perturbation, agent 1"),
        (axes[1], f0, f0 + 1, "shared force disturbance"),
    ]
    for ax, xidx, yidx, title in specs:
        ax.scatter(train_z[:, xidx], train_z[:, yidx], s=16, alpha=0.80, label="train")
        ax.scatter(test_z[:, xidx], test_z[:, yidx], s=16, alpha=0.45, label="test")
        if ref_z is not None:
            ax.scatter(ref_z[:, xidx], ref_z[:, yidx], s=10, alpha=0.30, label="reference")
        if cfg.show_titles:
            ax.set_title(title)
        ax.set_xlabel("x component")
        ax.set_ylabel("y component")
        _clean_axis(ax, cfg)
        _legend(ax)
    fig.tight_layout()
    _savefig(fig, path, cfg)


def write_metrics_csv(metrics_by_name: Dict[str, Dict], path: str | Path) -> None:
    scalar_keys = [
        "mean_cost", "median_cost", "q90_cost", "q95_cost", "max_cost",
        "collisions", "collision_rate", "final_mean_target_distance",
        "final_q90_target_distance", "min_interagent_distance", "q05_min_interagent_distance",
        "mean_control_norm", "max_control_norm", "mean_obstacle_penalty_raw", "q95_obstacle_penalty_raw",
    ]
    extra = [k for m in metrics_by_name.values() for k, v in m.items() if np.isscalar(v) and k not in scalar_keys and not isinstance(v, str)]
    keys = [k for k in scalar_keys if any(k in m for m in metrics_by_name.values())] + sorted(set(extra))
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["controller", *keys])
        for name, metrics in metrics_by_name.items():
            writer.writerow([name, *[metrics.get(k, "") for k in keys]])


def plot_all_diagnostics(
    out_dir: str | Path,
    metrics_by_name: Dict[str, Dict],
    history_by_name: Dict[str, list] | None,
    xbar,
    loss_fn=None,
    min_dist: float = 1.0,
    prefix: str = "",
    plot_cfg: PlotConfig | None = None,
) -> None:
    """Create the standard diagnostic/thesis-figure set for one evaluation."""
    cfg = plot_cfg or PlotConfig()
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    write_metrics_csv(metrics_by_name, out_dir / f"{prefix}metrics_summary.csv")

    if history_by_name:
        for name, hist in history_by_name.items():
            plot_loss_history(hist, plots / f"{prefix}loss_{name}.pdf", f"{name} training", cfg)
        plot_training_comparison(history_by_name, plots / f"{prefix}training_comparison.pdf", cfg)
        plot_lambda_history(history_by_name, plots / f"{prefix}lambda_vs_epoch.pdf", cfg)
        plot_lambda_search_objective(history_by_name, plots / f"{prefix}lambda_search_objective.pdf", cfg)

    for name, m in metrics_by_name.items():
        plot_trajectories(
            m["x_log"],
            xbar,
            plots / f"{prefix}trajectories_{name}.pdf",
            name if not prefix else f"{name} {prefix[:-1]}",
            loss_fn=loss_fn,
            min_dist=min_dist,
            costs=m.get("costs"),
            plot_cfg=cfg,
        )
        plot_worst_fraction_trajectories(
            m,
            xbar,
            plots / f"{prefix}trajectories_worst5pct_{name}.pdf",
            title=f"Worst {100 * cfg.worst_fraction:g}% trajectories: {name}",
            loss_fn=loss_fn,
            min_dist=min_dist,
            plot_cfg=cfg,
        )

    plot_trajectory_overlay(metrics_by_name, xbar, plots / f"{prefix}trajectories_overlay.pdf", loss_fn=loss_fn, min_dist=min_dist, plot_cfg=cfg)
    plot_cost_boxplot(metrics_by_name, plots / f"{prefix}cost_boxplot.pdf", cfg)
    plot_cost_histograms(metrics_by_name, plots / f"{prefix}cost_histogram.pdf", cfg)
    plot_cost_ecdf(metrics_by_name, plots / f"{prefix}cost_ecdf.pdf", cfg)
    plot_metrics_bar(metrics_by_name, plots / f"{prefix}metrics_bar.pdf", cfg)
    plot_target_distance(metrics_by_name, xbar, plots / f"{prefix}target_distance.pdf", cfg)
    plot_final_target_distance_boxplot(metrics_by_name, plots / f"{prefix}final_target_distance_boxplot.pdf", cfg)
    plot_interagent_distance(metrics_by_name, plots / f"{prefix}interagent_distance.pdf", min_dist=min_dist, plot_cfg=cfg)
    plot_min_interagent_boxplot(metrics_by_name, plots / f"{prefix}min_interagent_boxplot.pdf", min_dist=min_dist, plot_cfg=cfg)
    plot_control_norm(metrics_by_name, plots / f"{prefix}control_norm.pdf", cfg)
    plot_obstacle_penalty(metrics_by_name, plots / f"{prefix}obstacle_penalty_boxplot.pdf", cfg)
    plot_loss_components_bar(metrics_by_name, loss_fn, plots / f"{prefix}loss_components_bar.pdf", cfg)
    plot_loss_components_boxplot(metrics_by_name, loss_fn, plots / f"{prefix}loss_components_boxplot.pdf", cfg)
    plot_loss_components_over_time(metrics_by_name, loss_fn, plots, prefix=prefix, plot_cfg=cfg)
