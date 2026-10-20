import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import tempfile
import json

from safetensors import safe_open

from htl_package.configs import ModelConfig
from htl_package.checkpoint import save_checkpoint, load_checkpoint, load_model_for_inference
from htl_package.models import HTLRankingModel
from htl_package.constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS, DEVICE


class TestSaveLoadCheckpoint:
    def _make(self):
        model = HTLRankingModel(
            hidden_size=64, depth=2, dropout=0.1, ffn_hidden=32,
            extra_dim=EXTRA_DIM, global_dim=GLOBAL_DIM, num_tasks=NUM_TASKS,
        ).to(DEVICE)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, EXTRA_DIM))
        config = ModelConfig(hidden_size=64, depth=2, ffn_hidden=32)
        return model, scaler, config

    def test_save_creates_single_file(self):
        model, scaler, config = self._make()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir, model, scaler, config, "test_ckpt")
            base = Path(tmpdir) / "test_ckpt"
            # exactly one file, no pkl / json side-cars
            assert sorted(p.name for p in Path(tmpdir).iterdir()) == [
                "test_ckpt.safetensors"
            ]
            assert base.with_suffix(".safetensors").exists()
            assert not Path(str(base) + "_scaler.pkl").exists()
            assert not Path(str(base) + "_config.json").exists()

    def test_embedded_meta(self):
        model, scaler, config = self._make()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir, model, scaler, config, "test_ckpt")
            with safe_open(str(Path(tmpdir) / "test_ckpt.safetensors"),
                           framework="pt") as sf:
                meta = sf.metadata()
            assert meta["format"] == "htl-safetensors-v2"
            d = json.loads(meta["config"])
            assert d["hidden_size"] == 64
            assert d["depth"] == 2
            assert d["ffn_hidden"] == 32
            s = json.loads(meta["scaler"])
            assert s["n_features_in_"] == EXTRA_DIM
            assert len(s["mean_"]) == EXTRA_DIM
            assert len(s["scale_"]) == EXTRA_DIM

    def test_load_roundtrip(self):
        model, scaler, config = self._make()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir, model, scaler, config, "test_ckpt")
            state_dict, loaded_scaler, loaded_config = load_checkpoint(
                tmpdir, "test_ckpt"
            )
            assert isinstance(state_dict, dict)
            assert isinstance(loaded_scaler, StandardScaler)
            assert isinstance(loaded_config, ModelConfig)
            assert loaded_config.hidden_size == 64
            assert loaded_config.depth == 2
            # scaler round-trip
            np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_)
            np.testing.assert_allclose(loaded_scaler.scale_, scaler.scale_)
            probe = np.random.randn(4, EXTRA_DIM)
            np.testing.assert_allclose(
                loaded_scaler.transform(probe), scaler.transform(probe)
            )

    def test_load_model_for_inference(self):
        model, scaler, config = self._make()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir, model, scaler, config, "test_ckpt")
            loaded_model, loaded_scaler, loaded_config = load_model_for_inference(
                tmpdir, "test_ckpt"
            )
            assert isinstance(loaded_model, HTLRankingModel)
            assert not loaded_model.training
