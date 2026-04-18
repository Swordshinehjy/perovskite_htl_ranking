import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch, MagicMock

from htl_package.training import _metrics, _make_group_split
from htl_package.constants import TASK_NAMES


class TestMetrics:
    def test_perfect_ranking(self):
        n = 20
        s1 = np.array([[1.0]] * n)
        s2 = np.array([[0.0]] * n)
        y1 = np.array([[1.0]] * n)
        y2 = np.array([[0.0]] * n)
        m = _metrics(s1, s2, y1, y2)
        assert m["PCE_pair_acc"] == pytest.approx(1.0)

    def test_random_ranking(self):
        np.random.seed(42)
        n = 100
        s1 = np.random.randn(n, 1)
        s2 = np.random.randn(n, 1)
        y1 = np.random.randn(n, 1)
        y2 = np.random.randn(n, 1)
        m = _metrics(s1, s2, y1, y2)
        assert 0.0 <= m["PCE_pair_acc"] <= 1.0
        assert -1.0 <= m["PCE_spearman"] <= 1.0

    def test_spearman_perfect(self):
        n = 20
        s1 = np.linspace(0, 1, n).reshape(-1, 1)
        s2 = np.linspace(0, 0.5, n).reshape(-1, 1)
        y1 = np.linspace(0, 1, n).reshape(-1, 1)
        y2 = np.linspace(0, 0.5, n).reshape(-1, 1)
        m = _metrics(s1, s2, y1, y2)
        assert m["PCE_spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_equal_targets(self):
        import warnings
        from scipy import stats
        n = 10
        s1 = np.random.randn(n, 1)
        s2 = np.random.randn(n, 1)
        y1 = np.array([[5.0]] * n)
        y2 = np.array([[5.0]] * n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            m = _metrics(s1, s2, y1, y2)
        assert m["PCE_pair_acc"] == 0.0

    def test_output_keys(self):
        n = 10
        s1 = np.random.randn(n, 1)
        s2 = np.random.randn(n, 1)
        y1 = np.random.randn(n, 1)
        y2 = np.random.randn(n, 1)
        m = _metrics(s1, s2, y1, y2)
        assert "PCE_pair_acc" in m
        assert "PCE_spearman" in m


class TestMakeGroupSplit:
    def test_group_split_covers_all_indices(self):
        n = 100
        dois = np.random.choice(["doi_A", "doi_B", "doi_C", "doi_D", "doi_E"], n)
        df = pd.DataFrame({"doi": dois, "val": range(n)})
        tr_idx, va_idx, te_idx = _make_group_split(df, 0.2, 0.2, 42)
        all_idx = np.concatenate([tr_idx, va_idx, te_idx])
        assert len(set(all_idx)) == n

    def test_group_split_no_overlap(self):
        n = 100
        dois = np.random.choice(["doi_A", "doi_B", "doi_C", "doi_D", "doi_E"], n)
        df = pd.DataFrame({"doi": dois, "val": range(n)})
        tr_idx, va_idx, te_idx = _make_group_split(df, 0.2, 0.2, 42)
        assert len(set(tr_idx) & set(va_idx)) == 0
        assert len(set(tr_idx) & set(te_idx)) == 0
        assert len(set(va_idx) & set(te_idx)) == 0

    def test_group_split_same_doi_same_split(self):
        n = 100
        dois = np.array(["doi_A"] * 30 + ["doi_B"] * 30 + ["doi_C"] * 40)
        df = pd.DataFrame({"doi": dois, "val": range(n)})
        tr_idx, va_idx, te_idx = _make_group_split(df, 0.33, 0.33, 42)
        for idx_set in [tr_idx, va_idx, te_idx]:
            if len(idx_set) > 0:
                group_dois = set(dois[idx_set])
                for i in idx_set:
                    assert dois[i] in group_dois

    def test_group_split_reproducible(self):
        n = 100
        dois = np.random.choice(["doi_A", "doi_B", "doi_C", "doi_D"], n)
        df = pd.DataFrame({"doi": dois, "val": range(n)})
        tr1, va1, te1 = _make_group_split(df, 0.2, 0.2, 42)
        tr2, va2, te2 = _make_group_split(df, 0.2, 0.2, 42)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(va1, va2)
        np.testing.assert_array_equal(te1, te2)
