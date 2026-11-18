"""PyTorch MLP Candidate Ranker with Texture-Geometry Feature Coupling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import Polygon as ShapelyPolygon

from ..vision.vision_parser import evaluate_placement_texture_cost
from .geometry_kernel import PolygonCoords, bbox

FEATURE_DIM = 12


class CandidateRankerMLP(nn.Module):
    """Lightweight 3-layer MLP ranker scoring (part, anchor, rotation, texture) candidates."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Residual linear prior initialized with domain-informed weights
        self.linear_prior = nn.Linear(in_dim, 1, bias=False)
        self._init_prior()

    def _init_prior(self) -> None:
        with torch.no_grad():
            # Feature indices:
            # 0: norm_area (+), 1: convexity (+), 2: aspect (+), 3: grain_locked (+),
            # 4: norm_x (-), 5: norm_y (-), 6: bl_compactness (+), 7: defect_dist (small parts near defect, large away),
            # 8: contact_ratio (+), 9: pocket_fit (+), 10: color_cost (-), 11: vein_align (+)
            prior_w = torch.tensor(
                [[1.20, 0.35, 0.25, 0.40, -0.85, -0.95, 1.35, -0.30, 1.65, 1.10, -1.25, 1.40]],
                dtype=torch.float32,
            )
            self.linear_prior.weight.copy_(prior_w)
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.15)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear_prior(x) + 0.35 * self.net(x)).squeeze(-1)


def extract_candidate_features(
    item: dict[str, Any],
    poly: PolygonCoords,
    rotation_deg: float,
    placed_records: list[dict[str, Any]],
    obstacle_shps: list[ShapelyPolygon],
    width: float,
    height: float,
    texture_field: dict[str, Any] | None = None,
    use_texture_coupling: bool = True,
) -> np.ndarray:
    """Extract 12-D normalized geometric + texture-coupled feature vector."""
    min_x, min_y, max_x, max_y = bbox(poly)
    slab_area = max(1.0, width * height)
    diag = max(1.0, math.hypot(width, height))

    norm_area = float(item.get("area", 1000.0)) / slab_area
    convexity = float(item.get("convexity", 1.0))
    aspect = float(item.get("aspect_ratio", 1.0))
    grain_locked = 1.0 if item.get("grain_locked", False) else 0.0

    norm_x = min_x / max(1.0, width)
    norm_y = min_y / max(1.0, height)
    bl_compactness = 1.0 - (max_x / max(1.0, width) + max_y / max(1.0, height)) * 0.5

    if obstacle_shps:
        min_d = diag
        for obs in obstacle_shps:
            ox1, oy1, ox2, oy2 = obs.bounds
            dx = max(0.0, ox1 - max_x, min_x - ox2)
            dy = max(0.0, oy1 - max_y, min_y - oy2)
            d = math.hypot(dx, dy)
            if d < min_d:
                min_d = d
        min_def_dist = min_d / diag
    else:
        min_def_dist = 0.5

    # Small filler parts (< 4% of slab area) are encouraged to hug defect boundaries (sand-filling),
    # while large parts are encouraged to stay in clean regions
    if norm_area < 0.045:
        defect_prox_feature = 1.0 - min(1.0, min_def_dist * 4.0)
    else:
        defect_prox_feature = min(1.0, min_def_dist * 2.5)

    # Contact bonus with board walls, placed parts, and defect boundaries
    pw = max(1.0, max_x - min_x)
    ph = max(1.0, max_y - min_y)
    perim = 2.0 * (pw + ph)
    contact = 0.0
    if min_x <= 15.0 or abs(max_x - width) <= 15.0:
        contact += ph
    if min_y <= 15.0 or abs(max_y - height) <= 15.0:
        contact += pw

    for rec in placed_records:
        ox1, oy1, ox2, oy2 = rec["bbox"]
        near_x = abs(min_x - ox2) <= 12.0 or abs(max_x - ox1) <= 12.0
        ov_y = max(0.0, min(max_y, oy2) - max(min_y, oy1))
        near_y = abs(min_y - oy2) <= 12.0 or abs(max_y - oy1) <= 12.0
        ov_x = max(0.0, min(max_x, ox2) - max(min_x, ox1))
        if near_x and ov_y > 0:
            contact += ov_y
        if near_y and ov_x > 0:
            contact += ov_x

    contact_ratio = min(1.5, contact / perim)
    pocket_fit = min(1.0, float(item.get("area", 1000.0)) / max(1.0, pw * ph))

    if use_texture_coupling and texture_field is not None:
        tex_eval = evaluate_placement_texture_cost(
            poly_mm=poly,
            rotation_deg=rotation_deg,
            width_mm=width,
            height_mm=height,
            texture_field=texture_field,
            grain_locked=bool(item.get("grain_locked", False)),
            priority=int(item.get("priority", 1)),
        )
        color_cost = float(tex_eval["color_cost"])
        vein_err_rad = math.radians(float(tex_eval["vein_angle_error_deg"]))
        vein_align = math.cos(2.0 * vein_err_rad)
    else:
        color_cost = 0.0
        vein_align = 1.0

    return np.array(
        [
            norm_area,
            convexity,
            aspect,
            grain_locked,
            norm_x,
            norm_y,
            bl_compactness,
            defect_prox_feature,
            contact_ratio,
            pocket_fit,
            color_cost,
            vein_align,
        ],
        dtype=np.float32,
    )


_CACHED_RANKER: CandidateRankerMLP | None = None


def default_weights_path() -> Path:
    return Path(__file__).resolve().parents[2] / "experiments" / "results" / "mlp_ranker_weights.pt"


def load_or_init_ranker(weights_path: str | Path | None = None) -> CandidateRankerMLP:
    global _CACHED_RANKER
    path = Path(weights_path) if weights_path else default_weights_path()
    if _CACHED_RANKER is not None and weights_path is None:
        return _CACHED_RANKER

    torch.manual_seed(42)
    model = CandidateRankerMLP()
    if path.exists():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()
    if weights_path is None:
        _CACHED_RANKER = model
    return model


def train_mlp_ranker(
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    save_path: str | Path | None = None,
    epochs: int = 40,
    lr: float = 3e-3,
) -> dict[str, Any]:
    """Train the MLP Candidate Ranker using pairwise MarginRankingLoss on (winner, loser) feature pairs."""
    global _CACHED_RANKER
    torch.manual_seed(42)
    model = CandidateRankerMLP()
    model.train()

    if not training_pairs:
        # Generate synthetic domain-consistent preference pairs if empty
        rng = np.random.default_rng(42)
        for _ in range(400):
            f_win = rng.uniform(0.0, 1.0, size=(FEATURE_DIM,)).astype(np.float32)
            f_lose = f_win.copy()
            f_win[6] = rng.uniform(0.65, 0.98)  # high compactness
            f_win[8] = rng.uniform(0.55, 1.20)  # high contact
            f_win[10] = rng.uniform(0.02, 0.25)  # low color blemish cost
            f_win[11] = rng.uniform(0.75, 1.00)  # strong vein alignment
            f_lose[6] = rng.uniform(0.10, 0.45)
            f_lose[8] = rng.uniform(0.00, 0.30)
            f_lose[10] = rng.uniform(0.55, 0.95)
            f_lose[11] = rng.uniform(-0.80, 0.20)
            training_pairs.append((f_win, f_lose))

    x_win = torch.tensor(np.stack([p[0] for p in training_pairs]), dtype=torch.float32)
    x_lose = torch.tensor(np.stack([p[1] for p in training_pairs]), dtype=torch.float32)
    target = torch.ones(len(training_pairs), dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MarginRankingLoss(margin=0.25)

    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        s_win = model(x_win)
        s_lose = model(x_lose)
        loss = criterion(s_win, s_lose, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        acc = float((model(x_win) > model(x_lose)).float().mean().item())

    out_path = Path(save_path) if save_path else default_weights_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    _CACHED_RANKER = model

    return {
        "weights_path": str(out_path),
        "num_pairs": len(training_pairs),
        "epochs": epochs,
        "initial_loss": round(losses[0], 5),
        "final_loss": round(losses[-1], 5),
        "pairwise_accuracy": round(acc, 4),
    }


class RLPPOPolicyNet(nn.Module):
    """Domain-adapted Actor-Critic policy network for RL-PPO-Packer baseline (Yang et al., ACM TOG 2023).

    Operates on 10-D homogeneous geometric candidate descriptors (excluding SCTF color/vein features
    and Part-in-Part sink reuse) with an actor policy head and value critic head.
    """

    def __init__(self, in_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim // 2, 1)
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        self.geom_prior = nn.Linear(in_dim, 1, bias=False)
        with torch.no_grad():
            # Homogeneous geometric prior: area(+), convexity(+), aspect(+), grain(0), x(-), y(-), compact(+), def_dist(+), contact(+), pocket(+)
            w = torch.tensor([[1.15, 0.40, 0.20, 0.0, -0.90, -1.05, 1.30, 0.15, 1.45, 0.95]], dtype=torch.float32)
            self.geom_prior.weight.copy_(w)
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.15)
            nn.init.zeros_(self.actor_head.bias)
            nn.init.orthogonal_(self.critic_head.weight, gain=0.5)
            nn.init.zeros_(self.critic_head.bias)

    def forward(self, x_12d: torch.Tensor) -> torch.Tensor:
        x_geom = x_12d[..., :10]
        h = self.backbone(x_geom)
        logits = self.geom_prior(x_geom) + 0.30 * self.actor_head(h)
        return logits.squeeze(-1)

    def evaluate_actor_critic(self, x_12d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_geom = x_12d[..., :10]
        h = self.backbone(x_geom)
        logits = (self.geom_prior(x_geom) + 0.30 * self.actor_head(h)).squeeze(-1)
        value = self.critic_head(h).squeeze(-1)
        return logits, value


class GFPackScoreNet(nn.Module):
    """Domain-adapted Score/Gradient-Field network for GFPack++-Guided baseline (Xue et al., SIGGRAPH Asia 2023).

    Predicts conditional boundary-attraction potential and compactness gradient score over candidate poses.
    """

    def __init__(self, in_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.potential_prior = nn.Linear(in_dim, 1, bias=False)
        with torch.no_grad():
            # Gradient-field boundary attraction prior: emphasizes wall/part contact_ratio (idx 8) and bl_compactness (idx 6)
            w = torch.tensor([[1.05, 0.30, 0.25, 0.0, -1.10, -1.25, 1.40, 0.10, 1.55, 1.05]], dtype=torch.float32)
            self.potential_prior.weight.copy_(w)
            for m in self.score_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.15)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x_12d: torch.Tensor) -> torch.Tensor:
        x_geom = x_12d[..., :10]
        return (self.potential_prior(x_geom) + 0.30 * self.score_mlp(x_geom)).squeeze(-1)


_CACHED_RL_PPO: RLPPOPolicyNet | None = None
_CACHED_GFPACK: GFPackScoreNet | None = None


def rl_ppo_weights_path() -> Path:
    return Path(__file__).resolve().parents[2] / "experiments" / "results" / "rl_ppo_policy_weights.pt"


def gfpack_weights_path() -> Path:
    return Path(__file__).resolve().parents[2] / "experiments" / "results" / "gfpack_score_weights.pt"


def load_or_init_rl_ppo_policy(weights_path: str | Path | None = None) -> RLPPOPolicyNet:
    global _CACHED_RL_PPO
    path = Path(weights_path) if weights_path else rl_ppo_weights_path()
    if _CACHED_RL_PPO is not None and weights_path is None:
        return _CACHED_RL_PPO
    torch.manual_seed(101)
    model = RLPPOPolicyNet()
    if path.exists():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()
    if weights_path is None:
        _CACHED_RL_PPO = model
    return model


def load_or_init_gfpack_score_net(weights_path: str | Path | None = None) -> GFPackScoreNet:
    global _CACHED_GFPACK
    path = Path(weights_path) if weights_path else gfpack_weights_path()
    if _CACHED_GFPACK is not None and weights_path is None:
        return _CACHED_GFPACK
    torch.manual_seed(202)
    model = GFPackScoreNet()
    if path.exists():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()
    if weights_path is None:
        _CACHED_GFPACK = model
    return model

