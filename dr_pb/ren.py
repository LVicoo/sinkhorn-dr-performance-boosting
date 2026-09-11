from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContractiveREN(nn.Module):
    """A compact acyclic contractive REN.

    This is the only neural operator used in this implementation.  It is meant
    to be the IMC free operator M_theta in the performance-boosting controller.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_internal: int = 8,
        dim_nl: int = 8,
        internal_state_init: torch.Tensor | None = None,
        initialization_std: float = 0.1,
        pos_def_tol: float = 1e-3,
        contraction_rate_lb: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_nl = dim_nl
        self.epsilon = pos_def_tol
        self.contraction_rate_lb = contraction_rate_lb

        if internal_state_init is None:
            x0 = torch.zeros(1, 1, dim_internal)
        else:
            x0 = internal_state_init.reshape(1, 1, dim_internal).detach().clone()
        self.register_buffer("init_x", x0)
        self.x = self.init_x

        self.X_shape = (2 * dim_internal + dim_nl, 2 * dim_internal + dim_nl)
        self.Y_shape = (dim_internal, dim_internal)
        self.B2_shape = (dim_internal, dim_in)
        self.C2_shape = (dim_out, dim_internal)
        self.D21_shape = (dim_out, dim_nl)
        self.D22_shape = (dim_out, dim_in)
        self.D12_shape = (dim_nl, dim_in)
        self.training_param_names = ["X", "Y", "B2", "C2", "D21", "D22", "D12"]

        self._init_trainable_params(initialization_std)
        self.register_buffer("eye_mask_H", torch.eye(2 * dim_internal + dim_nl))
        self.register_buffer("eye_mask_w", torch.eye(dim_nl))

    def _init_trainable_params(self, initialization_std: float) -> None:
        for name in self.training_param_names:
            shape = getattr(self, f"{name}_shape")
            setattr(self, name, nn.Parameter(torch.randn(*shape) * initialization_std))

    def reset(self) -> None:
        self.x = self.init_x

    def _update_model_param(self) -> None:
        H = self.X.T @ self.X + self.epsilon * self.eye_mask_H
        h1, h2, h3 = torch.split(H, [self.dim_internal, self.dim_nl, self.dim_internal], dim=0)
        H11, _, _ = torch.split(h1, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)

        P = H33
        self.F = H31
        self.B1 = H32
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)
        self.E_inv = torch.linalg.inv(self.E)
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in: torch.Tensor) -> torch.Tensor:
        # u_in shape: [batch, 1, dim_in]
        self._update_model_param()
        batch_size = u_in.shape[0]
        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device, dtype=u_in.dtype)

        # Acyclic nonlinear block. Loop over dim_nl is intentional and small.
        for i in range(self.dim_nl):
            v = (
                F.linear(self.x, self.C1[i, :])
                + F.linear(w, self.D11[i, :])
                + F.linear(u_in, self.D12[i, :])
            )
            wi = torch.tanh(v / (self.Lambda[i] + 1e-8))
            w = w + (self.eye_mask_w[i, :] * wi).reshape(batch_size, 1, self.dim_nl)

        rhs = F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2)
        self.x = F.linear(rhs, self.E_inv)
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        return y_out

    def get_parameter_shapes(self):
        return OrderedDict((name, getattr(self, name).shape) for name in self.training_param_names)

    def get_named_parameters(self):
        return OrderedDict((name, getattr(self, name)) for name in self.training_param_names)
