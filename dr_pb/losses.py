from __future__ import annotations

import torch


class RobotsLoss:
    """Finite-horizon tracking/control/collision/obstacle loss.

    `per_sample` is the important method for DRO: it returns one scalar loss per
    rollout. `forward` returns the empirical mean and reproduces SAA training.
    """

    def __init__(
        self,
        xbar: torch.Tensor,
        Q: torch.Tensor | None = None,
        alpha_u: float = 0.1 / 400.0,
        alpha_col: float | None = 100.0,
        alpha_obst: float | None = None,
        n_agents: int = 2,
        min_dist: float = 1.0,
        obstacle_centers: list[torch.Tensor] | None = None,
        obstacle_covs: list[torch.Tensor] | None = None,
        loss_bound: float | None = None,
        sat_bound: float | None = None,
    ) -> None:
        self.xbar = xbar.reshape(1, 1, -1)
        state_dim = self.xbar.shape[-1]
        self.Q = torch.eye(state_dim, device=self.xbar.device) if Q is None else Q.to(self.xbar.device)
        self.alpha_u = alpha_u
        self.alpha_col = alpha_col
        self.alpha_obst = alpha_obst
        self.n_agents = n_agents
        self.min_dist = min_dist
        self.loss_bound = loss_bound
        self.sat_bound = sat_bound
        self.mask = torch.logical_not(torch.eye(n_agents, device=self.xbar.device))

        if obstacle_centers is None:
            obstacle_centers = [
                torch.tensor([[-2.5, 0.0]], device=self.xbar.device),
                torch.tensor([[2.5, 0.0]], device=self.xbar.device),
                torch.tensor([[-1.5, 0.0]], device=self.xbar.device),
                torch.tensor([[1.5, 0.0]], device=self.xbar.device),
            ]
        if obstacle_covs is None:
            obstacle_covs = [torch.tensor([[0.2, 0.2]], device=self.xbar.device)] * len(obstacle_centers)
        self.obstacle_centers = [c.to(self.xbar.device) for c in obstacle_centers]
        self.obstacle_covs = [c.to(self.xbar.device) for c in obstacle_covs]

    def per_sample(self, xs: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        # xs: [S,T,state_dim], us: [S,T,in_dim]
        x_centered = xs - self.xbar
        x_batch = x_centered.reshape(*x_centered.shape, 1)
        u_batch = us.reshape(*us.shape, 1)

        xTQx = torch.matmul(torch.matmul(x_batch.transpose(-1, -2), self.Q), x_batch).squeeze((-1, -2))
        loss_x = xTQx.mean(dim=1)

        uTRu = (u_batch.transpose(-1, -2) @ u_batch).squeeze((-1, -2))
        loss_u = self.alpha_u * uTRu.mean(dim=1)

        loss = loss_x + loss_u
        if self.alpha_col is not None and self.alpha_col > 0:
            loss = loss + self.alpha_col * self.collision_loss_per_sample(xs)
        if self.alpha_obst is not None and self.alpha_obst > 0:
            loss = loss + self.alpha_obst * self.obstacle_loss_per_sample(xs)

        if self.sat_bound is not None:
            loss = torch.tanh(loss / self.sat_bound)
        if self.loss_bound is not None:
            loss = self.loss_bound * loss
        return loss

    def forward(self, xs: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        return self.per_sample(xs, us).mean()

    __call__ = forward

    def collision_loss_per_sample(self, xs: torch.Tensor) -> torch.Tensor:
        distance_sq = self.get_pairwise_distance_sq(xs)
        min_sec_dist = self.min_dist + 0.2
        active = distance_sq.detach() < min_sec_dist**2
        # mask removes self-distance terms.
        penalty = (1.0 / (distance_sq + 1e-3)) * active * self.mask
        return penalty.sum(dim=(-1, -2)).mean(dim=1) / 2.0

    # Backward-compatible private name.
    def _collision_loss(self, xs: torch.Tensor) -> torch.Tensor:
        return self.collision_loss_per_sample(xs)

    def obstacle_loss_per_sample(self, xs: torch.Tensor) -> torch.Tensor:
        qx = xs[:, :, 0::4]
        qy = xs[:, :, 1::4]
        q = torch.cat((qx[..., None], qy[..., None]), dim=-1).reshape(xs.shape[0], xs.shape[1], -1)
        total = 0.0
        for center, cov in zip(self.obstacle_centers, self.obstacle_covs):
            total = total + normpdf_split_agents(q, center, cov)
        return total.mean(dim=1)

    # Backward-compatible private name.
    def _obstacle_loss(self, xs: torch.Tensor) -> torch.Tensor:
        return self.obstacle_loss_per_sample(xs)

    def get_pairwise_distance_sq(self, xs: torch.Tensor) -> torch.Tensor:
        if xs.ndim == 4:
            xs = xs.squeeze(-1)
        state_dim_per_agent = xs.shape[-1] // self.n_agents
        x_agents = xs[:, :, 0::state_dim_per_agent]
        y_agents = xs[:, :, 1::state_dim_per_agent]
        dx = x_agents.unsqueeze(-1) - x_agents.unsqueeze(-2)
        dy = y_agents.unsqueeze(-1) - y_agents.unsqueeze(-2)
        return dx**2 + dy**2

    def count_collisions(self, xs: torch.Tensor) -> float:
        dist_sq = self.get_pairwise_distance_sq(xs)
        col = (dist_sq > 1e-4) & (dist_sq < self.min_dist**2)
        return float(col.sum().detach().cpu().item() / 2.0)

    def obstacle_mahalanobis_sq(self, xs: torch.Tensor) -> torch.Tensor:
        """Return squared normalized distance to each obstacle.

        Shape: [samples, horizon, n_agents, n_obstacles].
        """
        if xs.ndim == 4:
            xs = xs.squeeze(-1)
        qx = xs[:, :, 0::4]
        qy = xs[:, :, 1::4]
        pos = torch.stack((qx, qy), dim=-1)  # [S,T,A,2]
        vals = []
        for center, cov in zip(self.obstacle_centers, self.obstacle_covs):
            center = center.reshape(1, 1, 1, 2)
            cov = cov.reshape(1, 1, 1, 2)
            vals.append(((pos - center) ** 2 / cov).sum(dim=-1))
        return torch.stack(vals, dim=-1)

    def count_obstacle_hits(self, xs: torch.Tensor, sigma_level: float = 2.0) -> float:
        if self.alpha_obst is None or self.alpha_obst <= 0:
            return 0.0
        mah_sq = self.obstacle_mahalanobis_sq(xs)
        hits = mah_sq <= sigma_level**2
        return float(hits.sum().detach().cpu().item())


def normpdf_split_agents(q: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    # q shape [S,T,2*n_agents], split into [S,T,2] per agent.
    d = 2
    mu = mu.reshape(1, 1, d)
    cov = cov.reshape(1, 1, d)
    den = (2 * torch.pi) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
    out = 0.0
    for qi in torch.split(q, 2, dim=-1):
        out = out + torch.exp((-0.5 * (qi - mu) ** 2 / cov).sum(-1)) / den
    return out
