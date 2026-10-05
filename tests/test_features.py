import pytest
import numpy as np
import pandas as pd

from htl_package.features import _extra_feat, _extra_feat_single, _global_feat
from htl_package.constants import EXTRA_COLS, GLOBAL_COLS, EXTRA_DIM, GLOBAL_DIM


def _make_pair_df(n=5):
    data = {}
    data["doi"] = ["test_doi"] * n
    data["MO_ITO"] = [1] * n
    for col in EXTRA_COLS:
        col1 = col.format(s="1")
        col2 = col.format(s="2")
        data[col1] = np.random.randn(n).astype(np.float32)
        data[col2] = np.random.randn(n).astype(np.float32)
    data["PCE_1"] = np.random.rand(n).astype(np.float32) * 25
    data["PCE_2"] = np.random.rand(n).astype(np.float32) * 25
    data["SMILES_1"] = ["CCO"] * n
    data["SMILES_2"] = ["CCN"] * n
    data["Material_1"] = [f"Mat1_{i}" for i in range(n)]
    data["Material_2"] = [f"Mat2_{i}" for i in range(n)]
    return pd.DataFrame(data)


def _make_single_df(n=5):
    data = {}
    data["Materials"] = [f"Mat_{i}" for i in range(n)]
    data["SMILES"] = ["CCO"] * n
    data["MO_ITO"] = [1] * n
    for col in EXTRA_COLS:
        col_clean = col.replace("_{s}", "")
        data[col_clean] = np.random.randn(n).astype(np.float32)
    return pd.DataFrame(data)


class TestExtraFeat:
    def test_shape(self):
        df = _make_pair_df(5)
        ef1 = _extra_feat(df, "1")
        ef2 = _extra_feat(df, "2")
        assert ef1.shape == (5, EXTRA_DIM)
        assert ef2.shape == (5, EXTRA_DIM)

    def test_dtype(self):
        df = _make_pair_df(5)
        ef1 = _extra_feat(df, "1")
        assert ef1.dtype == np.float32

    def test_fillna(self):
        df = _make_pair_df(5)
        col = EXTRA_COLS[0].format(s="1")
        df.loc[2, col] = np.nan
        ef1 = _extra_feat(df, "1")
        assert not np.isnan(ef1).any()

    def test_values_correct(self):
        df = _make_pair_df(3)
        ef1 = _extra_feat(df, "1")
        col0 = EXTRA_COLS[0].format(s="1")
        np.testing.assert_allclose(ef1[:, 0], df[col0].values.astype(np.float32))


class TestExtraFeatSingle:
    def test_shape(self):
        df = _make_single_df(5)
        ef = _extra_feat_single(df)
        assert ef.shape == (5, EXTRA_DIM)

    def test_dtype(self):
        df = _make_single_df(5)
        ef = _extra_feat_single(df)
        assert ef.dtype == np.float32

    def test_fillna(self):
        df = _make_single_df(5)
        col = EXTRA_COLS[0].replace("_{s}", "")
        df.loc[2, col] = np.nan
        ef = _extra_feat_single(df)
        assert not np.isnan(ef).any()


class TestGlobalFeat:
    def test_shape(self):
        df = _make_pair_df(5)
        gf = _global_feat(df)
        assert gf.shape == (5, GLOBAL_DIM)

    def test_dtype(self):
        df = _make_pair_df(5)
        gf = _global_feat(df)
        assert gf.dtype == np.float32

    def test_values_correct(self):
        df = _make_pair_df(3)
        gf = _global_feat(df)
        np.testing.assert_allclose(gf[:, 0], df["MO_ITO"].values.astype(np.float32))

    def test_fillna(self):
        df = _make_pair_df(5)
        df.loc[2, "MO_ITO"] = np.nan
        gf = _global_feat(df)
        assert not np.isnan(gf).any()
