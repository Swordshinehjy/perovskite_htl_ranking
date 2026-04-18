"""Feature extraction utilities for extra and global features."""

import numpy as np
import pandas as pd
from rdkit import Chem

from .constants import EXTRA_COLS, GLOBAL_COLS, logger


def _extra_feat(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Extract extra feature columns with the given suffix (e.g. '1' or '2')."""
    cols = [c.format(s=suffix) for c in EXTRA_COLS]
    return df[cols].fillna(0.0).values.astype(np.float32)


def _extra_feat_single(df: pd.DataFrame) -> np.ndarray:
    """Extract extra feature columns without suffix (single-molecule format)."""
    cols = [c.replace("_{s}", "") for c in EXTRA_COLS]
    return df[cols].fillna(0.0).values.astype(np.float32)


def _global_feat(df: pd.DataFrame) -> np.ndarray:
    """Extract global feature columns."""
    return df[GLOBAL_COLS].fillna(0.0).values.astype(np.float32)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and perform:
      - Drop invalid rows
    """
    df = pd.read_csv(csv_path)
    for s in ["1", "2"]:
        df[f"mol_{s}"] = df[f"SMILES_{s}"].apply(Chem.MolFromSmiles)
    df = df[df["mol_1"].notna() & df["mol_2"].notna()].reset_index(drop=True)
    logger.info(f"Preprocessed dataset: {len(df)} valid pairs")
    return df
