import pytest
from dataclasses import asdict

from htl_package.configs import (
    ModelConfig,
    TrainingConfig,
    FinetuneConfig,
    PredictConfig,
    SurrogateConfig,
)
from htl_package.constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.hidden_size == 300
        assert cfg.depth == 6
        assert cfg.dropout == 0.1
        assert cfg.ffn_hidden == 256
        assert cfg.extra_dim == EXTRA_DIM
        assert cfg.global_dim == GLOBAL_DIM
        assert cfg.num_tasks == NUM_TASKS
        assert cfg.aggregation == "mean"

    def test_custom_values(self):
        cfg = ModelConfig(hidden_size=128, depth=3, dropout=0.2)
        assert cfg.hidden_size == 128
        assert cfg.depth == 3
        assert cfg.dropout == 0.2

    def test_to_dict(self):
        cfg = ModelConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_size"] == 300
        assert d["depth"] == 6
        assert "extra_dim" in d

    def test_from_dict(self):
        d = {"hidden_size": 128, "depth": 3, "dropout": 0.2, "ffn_hidden": 64}
        cfg = ModelConfig.from_dict(d)
        assert cfg.hidden_size == 128
        assert cfg.depth == 3
        assert cfg.dropout == 0.2
        assert cfg.ffn_hidden == 64

    def test_from_dict_ignores_unknown_keys(self):
        d = {"hidden_size": 128, "unknown_key": 999}
        cfg = ModelConfig.from_dict(d)
        assert cfg.hidden_size == 128
        assert not hasattr(cfg, "unknown_key")

    def test_to_dict_roundtrip(self):
        cfg = ModelConfig(hidden_size=200, depth=4)
        d = cfg.to_dict()
        cfg2 = ModelConfig.from_dict(d)
        assert cfg == cfg2


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.csv_path == "htl-data-combinations.csv"
        assert cfg.save_dir == "checkpoints"
        assert cfg.epochs == 1000
        assert cfg.batch_size == 32
        assert cfg.lr == 5e-4
        assert cfg.weight_decay == 1e-5
        assert cfg.patience == 50
        assert cfg.early_stop_warmup == 20
        assert cfg.val_ratio == 0.1
        assert cfg.test_ratio == 0.1
        assert cfg.margin == 0.2
        assert cfg.seed == 42
        assert cfg.split == "random"
        assert cfg.n_cv_folds is None

    def test_custom_values(self):
        cfg = TrainingConfig(epochs=100, lr=1e-3, split="group", n_cv_folds=5)
        assert cfg.epochs == 100
        assert cfg.lr == 1e-3
        assert cfg.split == "group"
        assert cfg.n_cv_folds == 5


class TestFinetuneConfig:
    def test_defaults(self):
        cfg = FinetuneConfig()
        assert cfg.csv_path == "htl-data-combinations.csv"
        assert cfg.checkpoint_dir == "checkpoints"
        assert cfg.checkpoint_name == "best_model"
        assert cfg.save_dir == "checkpoints"
        assert cfg.finetune_epochs == 10
        assert cfg.batch_size == 32
        assert cfg.lr == 1e-5
        assert cfg.weight_decay == 1e-6
        assert cfg.margin == 0.2
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = FinetuneConfig(finetune_epochs=20, lr=5e-5)
        assert cfg.finetune_epochs == 20
        assert cfg.lr == 5e-5


class TestPredictConfig:
    def test_defaults(self):
        cfg = PredictConfig()
        assert cfg.predict_csv == ""
        assert cfg.checkpoint_dir == "checkpoints"
        assert cfg.checkpoint_name == "best_model"
        assert cfg.output_path == "predictions.csv"

    def test_custom_values(self):
        cfg = PredictConfig(predict_csv="test.csv", output_path="out.csv")
        assert cfg.predict_csv == "test.csv"
        assert cfg.output_path == "out.csv"


class TestSurrogateConfig:
    def test_defaults(self):
        cfg = SurrogateConfig()
        assert cfg.hidden_size == 128
        assert cfg.depth == 3
        assert cfg.dropout == 0.10
        assert cfg.ffn_hidden == 64
        assert cfg.aggregation == "mean"
        assert cfg.epochs == 300
        assert cfg.batch_size == 64
        assert cfg.lr == 1e-3
        assert cfg.weight_decay == 1e-5
        assert cfg.patience == 40
        assert cfg.val_ratio == 0.15
        assert cfg.seed == 42
