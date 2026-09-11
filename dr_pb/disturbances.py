from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DisturbanceSpec:
    n_agents: int = 2
    horizon: int = 100
    h: float = 0.05
    mass: float = 1.0
    std_pos: float = 0.2
    std_force: float = 0.0
    force_scale: float = 1.0
    # Nominal initial position offset relative to xbar. Length 2*n_agents.
    # If None, initial perturbations are centered at xbar.
    base_position_offset: tuple[float, ...] | None = None

    @property
    def state_dim(self) -> int:
        return 4 * self.n_agents

    @property
    def latent_dim(self) -> int:
        # position offsets for every agent/axis + shared force [fx, fy]
        return 2 * self.n_agents + 2

    def scale_vector(self, device: torch.device | str = "cpu") -> torch.Tensor:
        # Used to normalize the latent transport cost.
        pos_scale = max(float(self.std_pos), 1e-6)
        force_scale = max(float(self.std_force), 1e-6)
        scales = [pos_scale] * (2 * self.n_agents) + [force_scale] * 2
        return torch.tensor(scales, device=device, dtype=torch.float32)


class LatentDisturbanceDataset(Dataset):
    """Dataset of low-dimensional disturbance parameters z.

    z = [delta_position_agent_1, ..., delta_position_agent_N, shared_force_xy].
    Initial perturbations affect only positions. Force is shared across agents
    and applied to velocity coordinates at every time step.
    """

    def __init__(
        self,
        num_samples: int,
        spec: DisturbanceSpec,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.spec = spec
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        z = torch.zeros(num_samples, spec.latent_dim)
        z[:, : 2 * spec.n_agents] = spec.std_pos * torch.randn(num_samples, 2 * spec.n_agents, generator=gen)
        z[:, 2 * spec.n_agents :] = spec.std_force * torch.randn(num_samples, 2, generator=gen)
        self.z = z.to(device)

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.z[idx]

    def as_tensor(self) -> torch.Tensor:
        return self.z

    def trajectories(self) -> torch.Tensor:
        return expand_latent_to_trajectory(self.z, self.spec)


class ReferenceSampler:
    """Gaussian reference sampler ν over latent disturbances."""

    def __init__(self, spec: DisturbanceSpec, device: torch.device | str = "cpu", seed: int = 1234) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def sample(self, num_samples: int) -> torch.Tensor:
        z = torch.zeros(num_samples, self.spec.latent_dim)
        z[:, : 2 * self.spec.n_agents] = self.spec.std_pos * torch.randn(
            num_samples, 2 * self.spec.n_agents, generator=self.gen
        )
        z[:, 2 * self.spec.n_agents :] = self.spec.std_force * torch.randn(num_samples, 2, generator=self.gen)
        return z.to(self.device)


def expand_latent_to_trajectory(z: torch.Tensor, spec: DisturbanceSpec) -> torch.Tensor:
    """Map latent disturbance z to process-noise trajectory w[0:T].

    Initial position perturbation enters at t=0 only. Constant force enters the
    velocity coordinates for every t. The force is shared by all agents.
    """
    batch = z.shape[0]
    w = torch.zeros(batch, spec.horizon, spec.state_dim, device=z.device, dtype=z.dtype)

    # Initial position perturbation only at t = 0.
    pos_offsets = z[:, : 2 * spec.n_agents].reshape(batch, spec.n_agents, 2)
    if spec.base_position_offset is not None:
        base = torch.tensor(spec.base_position_offset, device=z.device, dtype=z.dtype).reshape(1, spec.n_agents, 2)
        pos_offsets = pos_offsets + base
    for agent in range(spec.n_agents):
        w[:, 0, 4 * agent + 0] = pos_offsets[:, agent, 0]
        w[:, 0, 4 * agent + 1] = pos_offsets[:, agent, 1]

    # Shared force for all agents, applied as acceleration contribution to v.
    force = spec.force_scale * z[:, 2 * spec.n_agents : 2 * spec.n_agents + 2]
    dv = spec.h / spec.mass * force
    for agent in range(spec.n_agents):
        w[:, :, 4 * agent + 2] += dv[:, 0:1]
        w[:, :, 4 * agent + 3] += dv[:, 1:2]
    return w


def normalize_latent(z: torch.Tensor, spec: DisturbanceSpec) -> torch.Tensor:
    return z / spec.scale_vector(z.device)


def pairwise_transport_cost(z_hat: torch.Tensor, z_ref: torch.Tensor, spec: DisturbanceSpec, normalize: bool = True) -> torch.Tensor:
    """Return c(z_hat_i,z_ref_k)=0.5||z_hat_i-z_ref_k||^2, shape [B,K]."""
    if normalize:
        z_hat = normalize_latent(z_hat, spec)
        z_ref = normalize_latent(z_ref, spec)
    diff = z_hat[:, None, :] - z_ref[None, :, :]
    return 0.5 * torch.sum(diff * diff, dim=-1)
