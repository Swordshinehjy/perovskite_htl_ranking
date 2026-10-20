"""Checkpoint save/load utilities and model inference helper.

Checkpoints are stored as a single ``.safetensors`` file with embedded
JSON metadata — no pickle is involved anywhere:

  {checkpoint_name}.safetensors
    ├─ tensors  : model weights (safetensors, safe & fast)
    └─ metadata (all strings):
         format  : "htl-safetensors-v2"
         config  : ModelConfig JSON
         scaler  : StandardScaler attributes JSON
                   (mean_ / scale_ / var_ / n_features_in_ / ...)
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file, load_file
from typing import Dict, Tuple

from .constants import logger, DEVICE
from .configs import ModelConfig
from .models import HTLRankingModel

#: Format tag stored in the safetensors metadata header.
CHECKPOINT_FORMAT = "htl-safetensors-v2"


def _scaler_to_dict(scaler: StandardScaler) -> Dict:
    """Serialize a fitted StandardScaler to plain JSON-safe structures."""
    d = {
        "n_features_in_": int(scaler.n_features_in_),
        "mean_": [float(x) for x in np.asarray(scaler.mean_).ravel()],
        "scale_": [float(x) for x in np.asarray(scaler.scale_).ravel()],
        "var_": [float(x) for x in np.asarray(scaler.var_).ravel()],
    }
    if hasattr(scaler, "n_samples_seen_"):
        d["n_samples_seen_"] = float(scaler.n_samples_seen_)
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        d["feature_names_in_"] = [str(x) for x in names]
    return d


def _scaler_from_dict(d: Dict) -> StandardScaler:
    """Rebuild a fitted StandardScaler from its JSON-safe attributes."""
    scaler = StandardScaler()
    scaler.n_features_in_ = int(d["n_features_in_"])
    scaler.mean_ = np.asarray(d["mean_"], dtype=np.float64)
    scaler.scale_ = np.asarray(d["scale_"], dtype=np.float64)
    scaler.var_ = np.asarray(d["var_"], dtype=np.float64)
    scaler.n_samples_seen_ = float(d.get("n_samples_seen_", 1))
    if "feature_names_in_" in d:
        scaler.feature_names_in_ = np.asarray(d["feature_names_in_"], dtype=object)
    return scaler


def save_checkpoint(
    save_dir: str,
    model: nn.Module,
    scaler: StandardScaler,
    config: ModelConfig,
    checkpoint_name: str = "best_model",
):
    """
    Save a checkpoint as ONE safetensors file (no pickle).

    The single file contains:
      - the model weights (as tensors)
      - the model config (JSON, in the safetensors metadata header)
      - the fitted StandardScaler attributes (JSON, in the header)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"{checkpoint_name}.safetensors"

    metadata = {
        "format": CHECKPOINT_FORMAT,
        "config": json.dumps(config.to_dict()),
        "scaler": json.dumps(_scaler_to_dict(scaler)),
    }
    save_file(model.state_dict(), str(out_path), metadata=metadata)

    logger.info(f"Checkpoint saved -> {out_path} (single safetensors, with meta)")


def load_checkpoint(
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    device: torch.device = DEVICE,
) -> Tuple[Dict[str, torch.Tensor], StandardScaler, ModelConfig]:
    """
    Load a checkpoint from the single safetensors file.

    Returns
    -------
    state_dict : dict
    scaler : StandardScaler
    config : ModelConfig
    """
    ckpt_path = Path(checkpoint_dir) / f"{checkpoint_name}.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from safetensors import safe_open
    with safe_open(str(ckpt_path), framework="pt") as sf:
        metadata = sf.metadata() or {}

    state_dict = load_file(str(ckpt_path))
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    if "config" not in metadata or "scaler" not in metadata:
        raise ValueError(
            f"{ckpt_path} is not a '{CHECKPOINT_FORMAT}' checkpoint "
            "(missing embedded config/scaler meta). "
            "Legacy pkl-based checkpoints are no longer supported."
        )

    config = ModelConfig.from_dict(json.loads(metadata["config"]))
    scaler = _scaler_from_dict(json.loads(metadata["scaler"]))

    return state_dict, scaler, config


def load_model_for_inference(
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
) -> Tuple[HTLRankingModel, StandardScaler, ModelConfig]:
    """
    Convenience: load checkpoint, construct model, load weights, set eval mode.

    Returns (model, scaler, config) with model already on DEVICE and in eval mode.
    """
    state_dict, scaler, config = load_checkpoint(checkpoint_dir, checkpoint_name)
    model = HTLRankingModel(**config.to_dict()).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, config
