import pytest
import numpy as np
import torch
import torch.nn as nn

from htl_package.models import (
    MultiTaskRankingLoss,
    EarlyStopping,
)
from htl_package.constants import NUM_TASKS


class TestMultiTaskRankingLoss:
    def test_output_is_tuple(self):
        criterion = MultiTaskRankingLoss(margin=0.2)
        B = 4
        s1 = torch.randn(B, NUM_TASKS)
        s2 = torch.randn(B, NUM_TASKS)
        y1 = torch.randn(B, NUM_TASKS)
        y2 = torch.randn(B, NUM_TASKS)
        result = criterion(s1, s2, y1, y2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_is_tensor(self):
        criterion = MultiTaskRankingLoss(margin=0.2)
        B = 4
        s1 = torch.randn(B, NUM_TASKS)
        s2 = torch.randn(B, NUM_TASKS)
        y1 = torch.randn(B, NUM_TASKS)
        y2 = torch.randn(B, NUM_TASKS)
        loss, log = criterion(s1, s2, y1, y2)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_log_is_dict(self):
        criterion = MultiTaskRankingLoss(margin=0.2)
        B = 4
        s1 = torch.randn(B, NUM_TASKS)
        s2 = torch.randn(B, NUM_TASKS)
        y1 = torch.randn(B, NUM_TASKS)
        y2 = torch.randn(B, NUM_TASKS)
        _, log = criterion(s1, s2, y1, y2)
        assert isinstance(log, dict)
        assert "total" in log
        assert "PCE_rank" in log
        assert "PCE_reg" in log

    def test_loss_nonnegative(self):
        criterion = MultiTaskRankingLoss(margin=0.2)
        B = 8
        s1 = torch.randn(B, NUM_TASKS)
        s2 = torch.randn(B, NUM_TASKS)
        y1 = torch.randn(B, NUM_TASKS)
        y2 = torch.randn(B, NUM_TASKS)
        loss, _ = criterion(s1, s2, y1, y2)
        assert loss.item() >= 0

    def test_zero_loss_when_correctly_ranked_large_margin(self):
        criterion = MultiTaskRankingLoss(margin=0.2, rank_weight=1.0, reg_weight=0.0)
        B = 4
        s1 = torch.tensor([[2.0]] * B)
        s2 = torch.tensor([[0.0]] * B)
        y1 = torch.tensor([[2.0]] * B)
        y2 = torch.tensor([[0.0]] * B)
        loss, log = criterion(s1, s2, y1, y2)
        assert log["PCE_rank"] == pytest.approx(0.0, abs=1e-6)

    def test_margin_effect(self):
        criterion_small = MultiTaskRankingLoss(margin=0.01)
        criterion_large = MultiTaskRankingLoss(margin=1.0)
        B = 4
        s1 = torch.tensor([[0.1]] * B)
        s2 = torch.tensor([[0.0]] * B)
        y1 = torch.tensor([[1.0]] * B)
        y2 = torch.tensor([[0.0]] * B)
        loss_small, _ = criterion_small(s1, s2, y1, y2)
        loss_large, _ = criterion_large(s1, s2, y1, y2)
        assert loss_large.item() > loss_small.item()

    def test_equal_targets_zero_rank_loss(self):
        criterion = MultiTaskRankingLoss(margin=0.2)
        B = 4
        s1 = torch.randn(B, NUM_TASKS)
        s2 = torch.randn(B, NUM_TASKS)
        y1 = torch.tensor([[5.0]] * B)
        y2 = torch.tensor([[5.0]] * B)
        _, log = criterion(s1, s2, y1, y2)
        assert log["PCE_rank"] == pytest.approx(0.0, abs=1e-6)

    def test_custom_weights(self):
        criterion = MultiTaskRankingLoss(
            margin=0.2, rank_weight=0.7, reg_weight=0.3
        )
        assert criterion.alpha == 0.7
        assert criterion.beta == 0.3


class TestEarlyStopping:
    def test_no_stop_initially(self):
        stopper = EarlyStopping(patience=5)
        model = nn.Linear(10, 5)
        assert stopper.step(1.0, model) is False

    def test_stop_after_patience(self):
        stopper = EarlyStopping(patience=3, delta=0.0)
        model = nn.Linear(10, 5)
        stopper.step(1.0, model)
        stopper.step(1.1, model)
        stopper.step(1.2, model)
        assert stopper.step(1.3, model) is True

    def test_reset_on_improvement(self):
        stopper = EarlyStopping(patience=3, delta=0.0)
        model = nn.Linear(10, 5)
        stopper.step(1.0, model)
        stopper.step(1.1, model)
        stopper.step(0.5, model)
        assert stopper.counter == 0

    def test_best_state_saved(self):
        stopper = EarlyStopping(patience=5, delta=0.0)
        model = nn.Linear(10, 5)
        stopper.step(1.0, model)
        assert stopper.best_state is not None
        for k, v in stopper.best_state.items():
            assert isinstance(v, torch.Tensor)

    def test_warmup_prevents_early_stop(self):
        stopper = EarlyStopping(patience=2, delta=0.0, warmup=5)
        model = nn.Linear(10, 5)
        for i in range(5):
            assert stopper.step(10.0, model) is False
        assert stopper.best_state is None

    def test_warmup_then_normal(self):
        stopper = EarlyStopping(patience=2, delta=0.0, warmup=3)
        model = nn.Linear(10, 5)
        for i in range(3):
            stopper.step(1.0, model)
        assert stopper.step(1.0, model) is False
        assert stopper.step(1.0, model) is False
        assert stopper.step(1.0, model) is True

    def test_delta_threshold(self):
        stopper = EarlyStopping(patience=2, delta=0.1)
        model = nn.Linear(10, 5)
        stopper.step(1.0, model)
        assert stopper.step(0.95, model) is False
        assert stopper.counter == 1

    def test_best_loss_updated(self):
        stopper = EarlyStopping(patience=5, delta=0.0)
        model = nn.Linear(10, 5)
        stopper.step(1.0, model)
        assert stopper.best_loss == 1.0
        stopper.step(0.5, model)
        assert stopper.best_loss == 0.5
