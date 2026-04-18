"""Neural network models for HTL ranking and surrogate prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

from chemprop.data.collate import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation, SumAggregation, NormAggregation

from .constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS, TASK_NAMES, DEVICE


class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) encoder.
    chemprop v2 version directly uses BondMessagePassing module.
    Output: [B, hidden_size]
    """

    def __init__(self,
                 hidden_size: int = 300,
                 depth: int = 3,
                 dropout: float = 0.1,
                 aggregation: str = "mean"):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.aggregation = aggregation

        self.mpnn = BondMessagePassing(
            d_h=hidden_size,
            depth=depth,
            dropout=dropout,
        )
        if aggregation == "mean":
            self.agg = MeanAggregation()
        elif aggregation == "sum":
            self.agg = SumAggregation()
        else:
            self.agg = NormAggregation()

    def forward(self, batch: BatchMolGraph) -> torch.Tensor:
        batch.to(DEVICE)
        H = self.mpnn(batch)
        mol_vecs = self.agg(H, batch.batch)
        return mol_vecs


class HTLRankingModel(nn.Module):
    """
    Architecture:
        SMILES  ──► D-MPNN ──► mol_emb [H]
                                        ├─ cat ──► FFN ──► [score_e, score_h]
        extra_features [E] ─────────────┤
        global_features [G] ────────────┘

    Two HTL materials share the same parameters (Siamese network).
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
        ffn_hidden: int = 256,
        extra_dim: int = EXTRA_DIM,
        global_dim: int = GLOBAL_DIM,
        num_tasks: int = NUM_TASKS,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.mpnn = DMPNNEncoder(hidden_size, depth, dropout, aggregation)

        ffn_in = hidden_size + extra_dim + global_dim
        self.ffn = nn.Sequential(
            nn.Linear(ffn_in, ffn_hidden),
            nn.LayerNorm(ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, ffn_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(ffn_hidden // 2, num_tasks),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self,
        mol_graphs: BatchMolGraph,
        extra: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, num_tasks] score vector."""
        emb = self.mpnn(mol_graphs)  # [B, H]
        x = torch.cat([emb, extra, global_feat], dim=-1)  # [B, H+E+G]
        return self.ffn(x)  # [B, T]

    def forward(
        self,
        mg1: BatchMolGraph,
        ef1: torch.Tensor,
        mg2: BatchMolGraph,
        ef2: torch.Tensor,
        gf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores_1, scores_2), each [B, T]."""
        return self.encode(mg1, ef1, gf), self.encode(mg2, ef2, gf)


class MultiTaskRankingLoss(nn.Module):
    """
    For task PCE compute:
      1. Margin Ranking Loss:
           L_rank = ReLU( margin - sign(y1 - y2) * (s1 - s2) )
      2. Delta Regression Loss:
           L_reg  = MSE( s1 - s2,  y1 - y2 )

    Total loss = α · L_rank + β · L_reg
    """

    def __init__(
        self,
        margin: float = 0.2,
        rank_weight: float = 0.6,
        reg_weight: float = 0.4,
        task_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.margin = margin
        self.alpha = rank_weight
        self.beta = reg_weight
        self.task_w = task_weights or [1.0] * NUM_TASKS

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=s1.device)
        log: Dict[str, float] = {}

        for t, (name, lam) in enumerate(zip(TASK_NAMES, self.task_w)):
            diff_s = s1[:, t] - s2[:, t]
            diff_y = y1[:, t] - y2[:, t]

            sign = diff_y.sign()

            nonzero_mask = sign != 0
            if nonzero_mask.any():
                sign_nonzero = sign[nonzero_mask]
                diff_s_nonzero = diff_s[nonzero_mask]
                l_rank = F.relu(self.margin -
                                sign_nonzero * diff_s_nonzero).mean()
            else:
                l_rank = torch.tensor(0.0, device=s1.device)

            l_reg = F.mse_loss(diff_s, diff_y)

            task_loss = lam * (self.alpha * l_rank + self.beta * l_reg)
            total += task_loss

            log[f"{name}_rank"] = l_rank.item()
            log[f"{name}_reg"] = l_reg.item()

        log["total"] = total.item()
        return total, log


class EarlyStopping:

    def __init__(self, patience: int = 50, delta: float = 1e-4, warmup: int = 0):
        """
        Parameters
        ----------
        patience : int
            How many consecutive epochs without improvement before stopping.
        delta : float
            Minimum decrease in val_loss to qualify as an improvement.
        warmup : int
            Number of initial epochs to skip before starting to track
            improvements.  During warmup the model trains freely; no
            best_state is saved and early stopping cannot trigger.
            This prevents a spurious low loss on tiny val sets (common
            with group / LOGO splits) from locking in a bad checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.warmup = warmup
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[Dict] = None
        self._epoch = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        self._epoch += 1

        # During warmup: train freely, don't save state or count.
        if self._epoch <= self.warmup:
            return False

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience


class SurrogateModel(nn.Module):
    """
    Lightweight surrogate GNN: D-MPNN -> FFN -> scalar
    Used to predict the value of a single extra feature, enabling chained IG attribution.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        depth:       int = 3,
        dropout:     float = 0.10,
        ffn_hidden:  int = 64,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.encoder = DMPNNEncoder(hidden_size, depth, dropout, aggregation)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.LayerNorm(ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, ffn_hidden // 2),
            nn.SiLU(),
            nn.Linear(ffn_hidden // 2, 1),
        )
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bmg: BatchMolGraph) -> torch.Tensor:
        """Returns [B, 1] scalar prediction."""
        emb = self.encoder(bmg)   # [B, H]
        return self.ffn(emb)      # [B, 1]
