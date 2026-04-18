"""Checkpoint save/load utilities and model inference helper."""

from pathlib import Path
import json

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file, load_file
import joblib
from typing import Dict, Tuple

from .constants import logger, DEVICE
from .configs import ModelConfig
from .models import HTLRankingModel


def save_checkpoint(
    save_dir: str,
    model: nn.Module,
    scaler: StandardScaler,
    config: ModelConfig,
    checkpoint_name: str = "best_model",
):
    """
    Save model checkpoint using safetensors format.

    Files created:
      - {checkpoint_name}.safetensors  : model weights (safe)
      - {checkpoint_name}_scaler.pkl   : sklearn scaler (requires trust)
      - {checkpoint_name}_config.json  : model config (safe)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_path = save_dir / checkpoint_name

    save_file(model.state_dict(), str(base_path.with_suffix(".safetensors")))

    joblib.dump(scaler, str(str(base_path) + "_scaler.pkl"))

    with open(str(base_path) + "_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info(f"Checkpoint saved → {base_path}.*")


def load_checkpoint(
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    device: torch.device = DEVICE,
) -> Tuple[Dict[str, torch.Tensor], StandardScaler, ModelConfig]:
    """
    Load model checkpoint from safetensors format.

    Returns
    -------
    state_dict : dict
    scaler : StandardScaler
    config : ModelConfig
    """
    base_path = Path(checkpoint_dir) / checkpoint_name

    state_dict = load_file(str(base_path.with_suffix(".safetensors")))
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    scaler = joblib.load(str(str(base_path) + "_scaler.pkl"))

    with open(str(base_path) + "_config.json", "r") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)

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
