from __future__ import annotations

import torch
import torch.nn as nn

from .ren import ContractiveREN


class PerfBoostController(nn.Module):
    """Internal-model performance-boosting controller with REN M_theta.

    At each step the controller reconstructs the process disturbance by
    comparing the measured state with the embedded noiseless plant model, then
    sends this reconstructed disturbance through the REN operator M_theta.
    """

    def __init__(
        self,
        noiseless_forward,
        input_init: torch.Tensor,
        output_init: torch.Tensor,
        dim_internal: int = 8,
        dim_nl: int = 8,
        output_amplification: float = 1.0,
        initialization_std: float = 0.1,
        pos_def_tol: float = 1e-3,
        contraction_rate_lb: float = 1.0,
    ) -> None:
        super().__init__()
        self.noiseless_forward = noiseless_forward
        self.input_init = input_init.reshape(1, 1, -1).detach().clone()
        self.output_init = output_init.reshape(1, 1, -1).detach().clone()
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]
        self.output_amplification = output_amplification

        self.emme = ContractiveREN(
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            dim_internal=dim_internal,
            dim_nl=dim_nl,
            initialization_std=initialization_std,
            pos_def_tol=pos_def_tol,
            contraction_rate_lb=contraction_rate_lb,
        )
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.last_input = self.input_init
        self.last_output = self.output_init
        self.emme.reset()

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        input_t = input_t.reshape(input_t.shape[0], 1, self.dim_in)
        u_noiseless = self.noiseless_forward(
            t=self.t,
            x=self.last_input,
            u=self.last_output,
        )
        w_hat = input_t - u_noiseless
        output = self.output_amplification * self.emme(w_hat)
        self.last_input = input_t
        self.last_output = output
        self.t += 1
        return output


class ZeroController(nn.Module):
    """Prestabilized baseline: no performance-boosting action, M(w)=0."""

    def __init__(self, dim_out: int) -> None:
        super().__init__()
        self.dim_out = dim_out

    def reset(self) -> None:
        pass

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_t.shape[0], 1, self.dim_out, device=input_t.device, dtype=input_t.dtype)
