from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobotsSystem(nn.Module):
    """Two-dimensional point-mass robots with pre-stabilizing spring/damper.

    State ordering per agent: [p_x, p_y, v_x, v_y].
    Input ordering per agent: [a_x, a_y].
    """

    def __init__(
        self,
        xbar: torch.Tensor,
        x_init: torch.Tensor | None = None,
        u_init: torch.Tensor | None = None,
        linear_plant: bool = False,
        k: float = 1.0,
        h: float = 0.05,
        mass: float = 1.0,
        damping: float = 1.0,
        nonlinear_drag: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear_plant = linear_plant
        self.h = h
        self.mass = mass
        self.k = k
        self.b = damping
        self.b2 = None if linear_plant else nonlinear_drag

        self.register_buffer("xbar", xbar.reshape(1, 1, -1).float())
        if x_init is None:
            x_init = self.xbar.detach().clone()
        self.register_buffer("x_init", x_init.reshape(1, 1, -1).float())

        self.n_agents = self.xbar.shape[-1] // 4
        self.state_dim = 4 * self.n_agents
        self.in_dim = 2 * self.n_agents
        if u_init is None:
            u_init = torch.zeros(1, 1, self.in_dim)
        self.register_buffer("u_init", u_init.reshape(1, 1, -1).float())

        B_single = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0 / mass, 0.0], [0.0, 1.0 / mass]]) * h
        self.register_buffer("B", torch.kron(torch.eye(self.n_agents), B_single).float())

        A1 = torch.eye(4 * self.n_agents)
        A2_single = torch.cat(
            (
                torch.cat((torch.zeros(2, 2), torch.eye(2)), dim=1),
                torch.cat(
                    (
                        torch.diag(torch.tensor([-k / mass, -k / mass])),
                        torch.diag(torch.tensor([-damping / mass, -damping / mass])),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        A2 = torch.kron(torch.eye(self.n_agents), A2_single)
        self.register_buffer("A_lin", (A1 + h * A2).float())
        self.register_buffer("mask", torch.tensor([[0.0, 0.0], [1.0, 1.0]]).repeat(self.n_agents, 1))

    def A_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.linear_plant
        # Adds speed-dependent drag on velocity coordinates.
        speed_like = torch.norm(x.view(-1, 2 * self.n_agents, 2) * self.mask, dim=-1, keepdim=True)
        speed_like = torch.kron(speed_like, torch.ones(2, 1, device=x.device, dtype=x.dtype))
        A3 = -self.b2 / self.mass * torch.diag_embed(speed_like.squeeze(-1), offset=0, dim1=-2, dim2=-1)
        return self.A_lin + self.h * A3

    def noiseless_forward(self, t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, self.state_dim)
        u = u.reshape(-1, 1, self.in_dim)
        if self.linear_plant:
            return F.linear(x - self.xbar, self.A_lin) + F.linear(u, self.B) + self.xbar
        return torch.bmm(x - self.xbar, self.A_nonlin(x).transpose(1, 2)) + F.linear(u, self.B) + self.xbar

    def forward(self, t: int, x: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.noiseless_forward(t, x, u) + w.reshape(-1, 1, self.state_dim)

    def rollout(self, controller, data: torch.Tensor, train: bool = False):
        """Roll out the closed loop under process disturbance data.

        data shape: [batch, horizon, state_dim].
        Returns x_log [batch,horizon,state_dim], None, u_log [batch,horizon,in_dim].
        """
        controller.reset()
        batch = data.shape[0]
        x = self.x_init.detach().clone().repeat(batch, 1, 1)
        u = self.u_init.detach().clone().repeat(batch, 1, 1)
        xs, us = [], []
        for t in range(data.shape[1]):
            x = self.forward(t=t, x=x, u=u, w=data[:, t : t + 1, :])
            u = controller(x)
            xs.append(x)
            us.append(u)
        controller.reset()
        x_log = torch.cat(xs, dim=1)
        u_log = torch.cat(us, dim=1)
        if not train:
            x_log = x_log.detach()
            u_log = u_log.detach()
        return x_log, None, u_log


def default_mountain_states(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return nominal target xbar and nominal initial state x0 for the mountain scenario."""
    x0 = torch.tensor([2.0, -2.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0], device=device)
    xbar = torch.tensor([-2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0], device=device)
    return xbar, x0
