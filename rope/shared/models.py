from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by LE-WM."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (T, B, D)
        """
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    """FeedForward network used in LE-WM transformers."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """LE-WM Transformer with optional AdaLN-zero blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type[nn.Module] = Block,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        smoothed_dim: int = 10,
        emb_dim: int = 10,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)


class MLPDynamicsPredictor(nn.Module):
    """Predict future embeddings from flattened embedding and action history."""

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        history_size: int,
        action_history_size: int | None = None,
        num_preds: int = 1,
        hidden_width: int = 1024,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if history_size < 1:
            raise ValueError("history_size must be positive.")
        if action_history_size is not None and action_history_size < 1:
            raise ValueError("action_history_size must be positive.")
        if num_preds < 1:
            raise ValueError("num_preds must be positive.")
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.action_history_size = int(action_history_size or history_size)
        self.num_preds = int(num_preds)
        input_dim = self.history_size * self.embed_dim + self.action_history_size * self.action_dim

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_width))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_width
        layers.append(nn.Linear(current_dim, self.num_preds * self.embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if emb.ndim != 3:
            raise ValueError(f"Expected emb with shape [batch, history, dim], got {emb.shape}.")
        if action.ndim != 3:
            raise ValueError(f"Expected action with shape [batch, action_history, dim], got {action.shape}.")
        if emb.shape[1] != self.history_size or action.shape[1] != self.action_history_size:
            raise ValueError(
                "History length mismatch: "
                f"emb={emb.shape[1]}, expected_emb={self.history_size}, "
                f"action={action.shape[1]}, expected_action={self.action_history_size}."
            )

        x = torch.cat((emb.flatten(1), action.flatten(1)), dim=-1)
        pred = self.net(x)
        return pred.reshape(emb.shape[0], self.num_preds, self.embed_dim)


def _build_koopman_encoder(
    input_dim: int,
    output_dim: int,
    *,
    hidden_width: int,
    depth: int,
    dropout: float = 0.0,
    activation_fn: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_width))
        layers.append(activation_fn())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_width
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DeepKoopmanNoDec(nn.Module):
    """Decoder-free Koopman model with lifted state z = [x, enc(x)]."""

    def __init__(
        self,
        *,
        state_dim: int,
        control_dim: int,
        embedding_dim: int,
        hidden_width: int = 1024,
        depth: int = 4,
        dropout: float = 0.0,
        activation_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        if control_dim < 1:
            raise ValueError("control_dim must be positive.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")

        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.embedding_dim = int(embedding_dim)
        self.latent_dim = self.state_dim + self.embedding_dim

        self.encoder = _build_koopman_encoder(
            self.state_dim,
            self.embedding_dim,
            hidden_width=hidden_width,
            depth=depth,
            dropout=dropout,
            activation_fn=activation_fn,
        )
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)

    def lift_state(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat((state, self.encoder(state)), dim=-1)

    def _rollout_latent(self, z_current: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        z_pred = []
        for step in range(control.shape[1]):
            z_current = self.A(z_current) + self.B(control[:, step])
            z_pred.append(z_current)
        return torch.stack(z_pred, dim=1)

    def forward(
        self,
        x_k: torch.Tensor,
        u_seq: torch.Tensor,
        x_next_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if x_k.ndim != 2:
            raise ValueError(f"Expected x_k with shape [batch, state_dim], got {x_k.shape}.")
        if u_seq.ndim != 3:
            raise ValueError(f"Expected u_seq with shape [batch, horizon, control_dim], got {u_seq.shape}.")
        if x_k.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {x_k.shape[-1]}.")
        if u_seq.shape[-1] != self.control_dim:
            raise ValueError(f"Expected control_dim={self.control_dim}, got {u_seq.shape[-1]}.")

        z_k = self.lift_state(x_k)
        z_pred_seq = self._rollout_latent(z_k, u_seq)
        x_pred_seq = z_pred_seq[..., : self.state_dim]
        if x_next_seq is None:
            return z_pred_seq

        if x_next_seq.ndim != 3:
            raise ValueError(
                f"Expected x_next_seq with shape [batch, horizon, state_dim], got {x_next_seq.shape}."
            )
        x_next_seq_flat = x_next_seq.reshape(-1, self.state_dim)
        z_target_seq = self.lift_state(x_next_seq_flat).reshape(x_next_seq.shape[0], x_next_seq.shape[1], -1)
        return z_pred_seq, x_pred_seq, z_target_seq


class DeepKoopmanLinDec(nn.Module):
    """Koopman model with nonlinear encoder and linear decoder x = C z."""

    def __init__(
        self,
        *,
        state_dim: int,
        control_dim: int,
        latent_dim: int,
        hidden_width: int = 1024,
        depth: int = 4,
        dropout: float = 0.0,
        activation_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        if control_dim < 1:
            raise ValueError("control_dim must be positive.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")

        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = _build_koopman_encoder(
            self.state_dim,
            self.latent_dim,
            hidden_width=hidden_width,
            depth=depth,
            dropout=dropout,
            activation_fn=activation_fn,
        )
        self.C = nn.Linear(self.latent_dim, self.state_dim, bias=True)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)

    def lift_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.C(latent)

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)

    def _rollout_latent(self, z_current: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        z_pred = []
        for step in range(control.shape[1]):
            z_current = self.A(z_current) + self.B(control[:, step])
            z_pred.append(z_current)
        return torch.stack(z_pred, dim=1)

    def forward(
        self,
        x_k: torch.Tensor,
        u_seq: torch.Tensor,
        x_next_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if x_k.ndim != 2:
            raise ValueError(f"Expected x_k with shape [batch, state_dim], got {x_k.shape}.")
        if u_seq.ndim != 3:
            raise ValueError(f"Expected u_seq with shape [batch, horizon, control_dim], got {u_seq.shape}.")
        if x_k.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {x_k.shape[-1]}.")
        if u_seq.shape[-1] != self.control_dim:
            raise ValueError(f"Expected control_dim={self.control_dim}, got {u_seq.shape[-1]}.")

        z_k = self.lift_state(x_k)
        z_pred_seq = self._rollout_latent(z_k, u_seq)
        x_pred_seq = self.decode(z_pred_seq)
        if x_next_seq is None:
            return z_pred_seq

        if x_next_seq.ndim != 3:
            raise ValueError(
                f"Expected x_next_seq with shape [batch, horizon, state_dim], got {x_next_seq.shape}."
            )
        x_next_seq_flat = x_next_seq.reshape(-1, self.state_dim)
        z_target_seq = self.lift_state(x_next_seq_flat).reshape(x_next_seq.shape[0], x_next_seq.shape[1], -1)
        return z_pred_seq, x_pred_seq, z_target_seq


class KoopmanDynamicsPredictor(DeepKoopmanNoDec):
    """Compatibility wrapper for one-step JEPA-style Koopman prediction."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        koopman_embed_dim: int,
        hidden_width: int = 1024,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            control_dim=action_dim,
            embedding_dim=koopman_embed_dim,
            hidden_width=hidden_width,
            depth=depth,
            dropout=dropout,
        )
        self.action_dim = self.control_dim
        self.koopman_embed_dim = self.embedding_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"Expected state with shape [batch, 1, dim], got {state.shape}.")
        if state.shape[1] != 1:
            raise ValueError(f"KoopmanDynamicsPredictor expects one initial state, got {state.shape[1]}.")
        return super().forward(state[:, 0], action)


class KoopmanLinearDecoderPredictor(DeepKoopmanLinDec):
    """Compatibility wrapper for one-step JEPA-style Koopman prediction."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        koopman_embed_dim: int,
        hidden_width: int = 1024,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            control_dim=action_dim,
            latent_dim=koopman_embed_dim,
            hidden_width=hidden_width,
            depth=depth,
            dropout=dropout,
        )
        self.action_dim = self.control_dim
        self.koopman_embed_dim = self.latent_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"Expected state with shape [batch, 1, dim], got {state.shape}.")
        if state.shape[1] != 1:
            raise ValueError(f"KoopmanLinearDecoderPredictor expects one initial state, got {state.shape[1]}.")
        return super().forward(state[:, 0], action)


class JEPA(nn.Module):
    """LE-WM joint-embedding predictive architecture."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info: dict[str, torch.Tensor], action_sequence: torch.Tensor, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        h = info["pixels"].size(2)
        b, s, t = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [h, t - h], dim=2)
        info["action"] = act_0
        n_steps = t - h

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)
        emb = info["emb"] = init["emb"].unsqueeze(1).expand(b, s, -1, -1)
        init = {k: detach_clone(v) for k, v in init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        hs = history_size
        for step in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -hs:]
            act_trunc = act_emb[:, -hs:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, step : step + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -hs:]
        act_trunc = act_emb[:, -hs:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=b, s=s)
        return info

    def criterion(self, info_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(self, info_dict: dict[str, torch.Tensor], action_candidates: torch.Tensor) -> torch.Tensor:
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for key in list(info_dict.keys()):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for key in list(info_dict.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)

        goal.pop("action")
        goal = self.encode(goal)
        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


def build_mlp(
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    depth: int,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be at least 1.")

    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.GELU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class ResNetBackbone(nn.Module):
    def __init__(self, block_counts: tuple[int, int, int, int] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.layer1 = self._make_layer(64, block_counts[0], stride=1)
        self.layer2 = self._make_layer(128, block_counts[1], stride=2)
        self.layer3 = self._make_layer(256, block_counts[2], stride=2)
        self.layer4 = self._make_layer(512, block_counts[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class ResNet18LatentEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 24,
        proj_hidden_dim: int = 512,
        proj_depth: int = 1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.backbone = ResNetBackbone(block_counts=(2, 2, 2, 2))
        self.projector = build_mlp(
            in_dim=self.backbone.out_dim,
            out_dim=latent_dim,
            hidden_dim=proj_hidden_dim,
            depth=proj_depth,
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.latent_dim = int(latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        latents = self.projector(features)
        return self.output_norm(latents)


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if state_dim % latent_dim != 0:
            raise ValueError("state_dim must be divisible by latent_dim.")

        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.net = build_mlp(
            in_dim=state_dim + action_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, action), dim=-1))


def make_history_states(frame_latents: torch.Tensor, history: int) -> torch.Tensor:
    if frame_latents.ndim != 3:
        raise ValueError("frame_latents must have shape [batch, time, latent_dim].")
    batch, total_frames, latent_dim = frame_latents.shape
    if total_frames < history:
        raise ValueError("Not enough frames to build history states.")

    windows = []
    for start in range(total_frames - history + 1):
        window = frame_latents[:, start : start + history].reshape(batch, history * latent_dim)
        windows.append(window)
    return torch.stack(windows, dim=1)


def make_initial_padded_history_states(
    frame_latents: torch.Tensor,
    history: int,
    *,
    horizon: int | None = None,
) -> torch.Tensor:
    if frame_latents.ndim != 3:
        raise ValueError("frame_latents must have shape [batch, time, latent_dim].")
    batch, total_frames, latent_dim = frame_latents.shape
    if history < 1:
        raise ValueError("history must be positive.")
    if total_frames < 1:
        raise ValueError("At least one frame is required to build padded history states.")

    horizon = total_frames - 1 if horizon is None else int(horizon)
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    if total_frames < horizon + 1:
        raise ValueError("Not enough frames to build padded history states for the requested horizon.")

    states = []
    for end in range(horizon + 1):
        start = max(0, end - history + 1)
        window = frame_latents[:, start : end + 1]
        pad_count = history - window.shape[1]
        if pad_count > 0:
            padding = frame_latents[:, :1].expand(batch, pad_count, latent_dim)
            window = torch.cat((padding, window), dim=1)
        states.append(window.reshape(batch, history * latent_dim))
    return torch.stack(states, dim=1)


def rollout_dynamics(
    dynamics: nn.Module,
    initial_state: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, horizon, action_dim].")

    predictions = []
    state = initial_state
    latent_dim = int(dynamics.latent_dim)
    for step in range(actions.shape[1]):
        current_latent = state[:, -latent_dim:]
        next_latent = current_latent + dynamics(state, actions[:, step])
        state = torch.cat((state[:, latent_dim:], next_latent), dim=-1)
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def symmetric_stopgrad_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = F.mse_loss(pred, target.detach())
    backward = F.mse_loss(pred.detach(), target)
    return 0.5 * (forward + backward)


def curvature_loss(frame_latents: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if frame_latents.shape[1] < 3:
        return frame_latents.new_zeros(())

    velocities = frame_latents[:, 1:] - frame_latents[:, :-1]
    v_t = velocities[:, :-1]
    v_next = velocities[:, 1:]
    numerator = (v_t * v_next).sum(dim=-1)
    denominator = v_t.norm(dim=-1).clamp_min(eps) * v_next.norm(dim=-1).clamp_min(eps)
    return (1.0 - numerator / denominator).mean()


@dataclass
class LatentDynamicsOutput:
    loss: torch.Tensor
    dynamics_loss: torch.Tensor
    curvature_loss: torch.Tensor
    encoded_latents: torch.Tensor
    predicted_states: torch.Tensor
    target_states: torch.Tensor


class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 24,
        history: int = 3,
        action_dim: int = 2,
        encoder_proj_hidden_dim: int = 512,
        encoder_proj_depth: int = 1,
        dynamics_hidden_dim: int = 1024,
        dynamics_depth: int = 3,
        curvature_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if history < 1:
            raise ValueError("history must be at least 1.")

        self.encoder = ResNet18LatentEncoder(
            latent_dim=latent_dim,
            proj_hidden_dim=encoder_proj_hidden_dim,
            proj_depth=encoder_proj_depth,
        )
        self.history = int(history)
        self.latent_dim = int(latent_dim)
        self.state_dim = int(history * latent_dim)
        self.action_dim = int(action_dim)
        self.dynamics = LatentDynamicsMLP(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            hidden_dim=dynamics_hidden_dim,
            depth=dynamics_depth,
        )
        self.curvature_weight = float(curvature_weight)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        latents = self.encoder(frames.reshape(batch * time, channels, height, width))
        return latents.reshape(batch, time, -1)

    def compute_losses(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rollout_start_idx: torch.Tensor | None = None,
    ) -> LatentDynamicsOutput:
        batch = latents.shape[0]
        horizon = actions.shape[1]
        if rollout_start_idx is None:
            rollout_start_idx = torch.zeros(batch, dtype=torch.long, device=latents.device)
        else:
            rollout_start_idx = rollout_start_idx.to(device=latents.device, dtype=torch.long)
        if rollout_start_idx.shape != (batch,):
            raise ValueError("rollout_start_idx must have shape [batch].")

        max_state_idx = int(rollout_start_idx.max().item()) + horizon
        states = make_initial_padded_history_states(latents, self.history, horizon=max_state_idx)
        batch_idx = torch.arange(batch, device=latents.device)
        target_offsets = rollout_start_idx[:, None] + torch.arange(1, horizon + 1, device=latents.device)

        initial = states[batch_idx, rollout_start_idx]
        target = states[batch_idx[:, None], target_offsets]

        pred = rollout_dynamics(self.dynamics, initial, actions)
        pred_next = pred[..., -self.latent_dim :]
        target_next = target[..., -self.latent_dim :]

        dynamics_loss = symmetric_stopgrad_mse(pred_next, target_next)

        curvature = latents.new_zeros(())
        if self.curvature_weight > 0.0:
            rollout_offsets = rollout_start_idx[:, None] + torch.arange(horizon + 1, device=latents.device)
            rollout_latents = latents[batch_idx[:, None], rollout_offsets]
            curvature = curvature_loss(rollout_latents)

        total_loss = dynamics_loss + self.curvature_weight * curvature
        return LatentDynamicsOutput(
            loss=total_loss,
            dynamics_loss=dynamics_loss,
            curvature_loss=curvature,
            encoded_latents=latents,
            predicted_states=pred,
            target_states=target,
        )

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        rollout_start_idx: torch.Tensor | None = None,
    ) -> LatentDynamicsOutput:
        latents = self.encode_frames(frames)
        return self.compute_losses(
            latents=latents,
            actions=actions,
            rollout_start_idx=rollout_start_idx,
        )


def create_koopman_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    depth: int,
    activation_fn: type[nn.Module] | str = nn.ReLU,
) -> nn.Sequential:
    if isinstance(activation_fn, str):
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        activation_fn = activations[activation_fn.lower()]

    if depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_width), activation_fn()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_width, hidden_width), activation_fn()])
    layers.append(nn.Linear(hidden_width, output_dim))
    return nn.Sequential(*layers)


class KoopmanEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: type[nn.Module] | str,
    ) -> None:
        super().__init__()
        self.net = create_koopman_mlp(input_dim, latent_dim, hidden_width, depth, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryDeepKoopmanNoDec(nn.Module):
    """Decoder-free Koopman model with lifted state z = [x_t, enc(history)]."""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        embedding_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: type[nn.Module] | str,
        history: int = 3,
    ) -> None:
        super().__init__()
        if history < 1:
            raise ValueError("history must be at least 1.")

        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.embedding_dim = int(embedding_dim)
        self.history = int(history)
        self.history_dim = self.history * self.state_dim
        self.latent_dim = self.state_dim + self.embedding_dim

        self.encoder = KoopmanEncoder(self.history_dim, embedding_dim, hidden_width, depth, activation_fn)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(control_dim, self.latent_dim, bias=False)

    def lift_state(self, history: torch.Tensor) -> torch.Tensor:
        """Lift chronological history [x_{t-h+1}, ..., x_t] into Koopman coordinates."""
        current_state = history[..., -self.state_dim :]
        return torch.cat([current_state, self.encoder(history)], dim=-1)

    def forward(
        self,
        history_k: torch.Tensor,
        u_seq: torch.Tensor,
        history_next_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        history_k: [B, H * STATE_DIM] chronological history ending at x_t
        u_seq: [B, M, CONTROL_DIM] actions a_t ... a_{t+M-1}
        history_next_seq: [B, M, H * STATE_DIM] histories ending at x_{t+1} ... x_{t+M}
        """
        batch, horizon = u_seq.shape[:2]

        z_k = self.lift_state(history_k)

        history_next_flat = history_next_seq.reshape(batch * horizon, self.history_dim)
        z_target_seq_flat = self.lift_state(history_next_flat)
        z_target_seq = z_target_seq_flat.reshape(batch, horizon, self.latent_dim)

        z_pred_list = []
        z_current_pred = z_k
        for step in range(horizon):
            u_i = u_seq[:, step, :]
            z_next_pred = self.A(z_current_pred) + self.B(u_i)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        x_pred_seq = z_pred_seq[..., : self.state_dim]
        return z_pred_seq, x_pred_seq, z_target_seq
