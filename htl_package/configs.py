"""Dataclass configurations for models and training."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from .constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_size: int = 300
    depth: int = 6
    dropout: float = 0.1
    ffn_hidden: int = 256
    extra_dim: int = EXTRA_DIM
    global_dim: int = GLOBAL_DIM
    num_tasks: int = NUM_TASKS
    aggregation: str = "mean"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{
            k: v
            for k, v in d.items() if k in cls.__dataclass_fields__
        })


@dataclass
class TrainingConfig:
    """Training configuration"""
    csv_path: str = "htl-data-combinations.csv"
    save_dir: str = "checkpoints"
    epochs: int = 1000
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 50
    early_stop_warmup: int = 20   # epochs to skip before early stopping starts
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    margin: float = 0.2
    seed: int = 42
    split: str = "random"           # "random" | "group"
    n_cv_folds: Optional[int] = None  # None = single split; int = LOGO CV top-N folds


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration"""
    csv_path: str = "htl-data-combinations.csv"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model"
    save_dir: str = "checkpoints"
    finetune_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-6
    margin: float = 0.2
    seed: int = 42


@dataclass
class PredictConfig:
    """Prediction configuration"""
    predict_csv: str = ""
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model"
    output_path: str = "predictions.csv"


@dataclass
class SurrogateConfig:
    """Hyperparameters: train a surrogate GNN for each extra feature"""
    hidden_size: int  = 128
    depth:       int  = 3
    dropout:     float = 0.10
    ffn_hidden:  int  = 64
    aggregation: str  = "mean"
    epochs:      int  = 300
    batch_size:  int  = 64
    lr:          float = 1e-3
    weight_decay: float = 1e-5
    patience:    int  = 40
    val_ratio:   float = 0.15
    seed:        int  = 42
