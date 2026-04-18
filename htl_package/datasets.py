"""Dataset classes and collate functions for HTL ranking and surrogate models."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional
from rdkit import Chem

from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer

from .constants import GLOBAL_COLS, GLOBAL_DIM, logger, DEVICE
from .features import _extra_feat, _extra_feat_single, _global_feat


def _make_featurizer():
    """Create a standard chemprop featurizer (shared across all datasets)."""
    return SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=MultiHotAtomFeaturizer.v2(),
        bond_featurizer=MultiHotBondFeaturizer(),
    )


# Pair Dataset


class PairDataset(Dataset):
    """
    Each sample = a pair of HTL materials (mol1, mol2)
    Contains:
      - MolGraph (chemprop molecular graph for D-MPNN)
      - extra_features (additional features, standardized)
      - global_features (global features, shared per pair)
      - targets [PCE] (for constructing ranking supervision)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.featurizer = _make_featurizer()
        self.mols1 = []
        self.mols2 = []
        self.graphs1 = []
        self.graphs2 = []
        valid_indices = []
        for idx in range(len(self.df)):
            mol1 = self.df.loc[idx, "mol_1"]
            mol2 = self.df.loc[idx, "mol_2"]
            g1 = self.featurizer(mol1) if mol1 else None
            g2 = self.featurizer(mol2) if mol2 else None
            if g1 is not None and g2 is not None:
                self.mols1.append(mol1)
                self.mols2.append(mol2)
                self.graphs1.append(g1)
                self.graphs2.append(g2)
                valid_indices.append(idx)

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        ef1 = _extra_feat(self.df, "1")
        ef2 = _extra_feat(self.df, "2")

        if fit_scaler:
            ef_all = np.vstack([ef1, ef2])
            self.scaler = StandardScaler().fit(ef_all)
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2

        self.gf = _global_feat(self.df)

        self.y1 = self.df[["PCE_1"]].values.astype(np.float32)
        self.y2 = self.df[["PCE_2"]].values.astype(np.float32)

    def _featurize(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.featurizer(mol)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.graphs1[idx],
            self.graphs2[idx],
            torch.tensor(self.ef1[idx]),
            torch.tensor(self.ef2[idx]),
            torch.tensor(self.gf[idx]),
            torch.tensor(self.y1[idx]),
            torch.tensor(self.y2[idx]),
        )


# List Dataset


class ListDataset(Dataset):
    """
    Single material list dataset for list ranking mode.
    Each sample = a single HTL material
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler,
        global_feat: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.featurizer = _make_featurizer()

        self.materials = []
        self.mols = []
        self.graphs = []
        valid_indices = []

        smiles_col = None
        for col in ["SMILES", "smiles", "Smiles"]:
            if col in self.df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError(
                "CSV must contain a SMILES column (SMILES/smiles/Smiles)")
        self.smiles_col = smiles_col

        materials_col = None
        for col in [
                "Materials", "materials", "Material", "material", "Name",
                "name"
        ]:
            if col in self.df.columns:
                materials_col = col
                break
        self.materials_col = materials_col

        for idx in range(len(self.df)):
            smiles = self.df.loc[idx, smiles_col]
            mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None
            g = self.featurizer(mol) if mol else None
            if g is not None:
                self.mols.append(mol)
                self.graphs.append(g)
                valid_indices.append(idx)
                if materials_col:
                    self.materials.append(str(self.df.loc[idx, materials_col]))
                else:
                    self.materials.append(f"Material_{idx}")

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.ef = self.scaler.transform(_extra_feat_single(self.df))

        if global_feat is not None:
            self.gf = np.tile(global_feat,
                              (len(self.df), 1)).astype(np.float32)
        elif all(col in self.df.columns for col in GLOBAL_COLS):
            self.gf = _global_feat(self.df)
        else:
            self.gf = np.zeros((len(self.df), GLOBAL_DIM), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            torch.tensor(self.ef[idx]),
            torch.tensor(self.gf[idx]),
            self.materials[idx],
        )


# Cached Pair Dataset


class CachedPairDataset:
    """
    Pre-built BatchMolGraph dataset, suitable for shuffle=False scenarios.
    Pre-builds all batch BatchMolGraphs during initialization to avoid repeated construction per epoch.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.featurizer = _make_featurizer()

        self.graphs1 = []
        self.graphs2 = []
        valid_indices = []
        for idx in range(len(self.df)):
            mol1 = self.df.loc[idx, "mol_1"]
            mol2 = self.df.loc[idx, "mol_2"]
            g1 = self.featurizer(mol1) if mol1 else None
            g2 = self.featurizer(mol2) if mol2 else None
            if g1 is not None and g2 is not None:
                self.graphs1.append(g1)
                self.graphs2.append(g2)
                valid_indices.append(idx)

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        ef1 = _extra_feat(self.df, "1")
        ef2 = _extra_feat(self.df, "2")

        if fit_scaler:
            ef_all = np.vstack([ef1, ef2])
            self.scaler = StandardScaler().fit(ef_all)
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2

        self.gf = _global_feat(self.df)

        self.y1 = self.df[["PCE_1"]].values.astype(np.float32)
        self.y2 = self.df[["PCE_2"]].values.astype(np.float32)

        self._build_cached_batches()

    def _build_cached_batches(self):
        """Pre-build all batch BatchMolGraphs."""
        self._cached_batches = []
        n = len(self.df)
        for i in range(0, n, self.batch_size):
            end_idx = min(i + self.batch_size, n)
            batch_indices = list(range(i, end_idx))

            g1s = [self.graphs1[idx] for idx in batch_indices]
            g2s = [self.graphs2[idx] for idx in batch_indices]

            bmg1 = BatchMolGraph(g1s)
            bmg2 = BatchMolGraph(g2s)

            ef1_batch = torch.tensor(self.ef1[batch_indices])
            ef2_batch = torch.tensor(self.ef2[batch_indices])
            gf_batch = torch.tensor(self.gf[batch_indices])
            y1_batch = torch.tensor(self.y1[batch_indices])
            y2_batch = torch.tensor(self.y2[batch_indices])

            self._cached_batches.append((bmg1, bmg2, ef1_batch, ef2_batch,
                                         gf_batch, y1_batch, y2_batch))

    def __len__(self):
        return len(self._cached_batches)

    def __getitem__(self, idx):
        return self._cached_batches[idx]


# Surrogate Dataset


class SurrogateDataset(Dataset):
    """(SMILES, float_target) dataset for training a single surrogate model."""

    def __init__(self, smiles_list, targets: np.ndarray):
        self.featurizer = _make_featurizer()
        self.graphs: list = []
        self.targets_list: list = []
        for smi, tgt in zip(smiles_list, targets):
            mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
            if mol is None:
                continue
            g = self.featurizer(mol)
            self.graphs.append(g)
            self.targets_list.append(float(tgt))
        self.targets_arr = np.array(self.targets_list, dtype=np.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor([self.targets_arr[idx]])


# Collate functions


def collate_list_fn(batch):
    graphs, efs, gfs, materials = zip(*batch)
    graphs = list(graphs)
    if any(g is None for g in graphs):
        raise ValueError("Found None in graphs")
    bmg = BatchMolGraph(graphs)
    return bmg, torch.stack(efs), torch.stack(gfs), list(materials)


def collate_fn(batch):
    g1s, g2s, ef1s, ef2s, gfs, y1s, y2s = zip(*batch)
    g1s = list(g1s)
    g2s = list(g2s)
    if any(g is None for g in g1s) or any(g is None for g in g2s):
        raise ValueError(
            f"Found None in graphs: g1s has {sum(1 for g in g1s if g is None)} Nones, g2s has {sum(1 for g in g2s if g is None)} Nones"
        )
    bmg1 = BatchMolGraph(g1s)
    bmg2 = BatchMolGraph(g2s)
    return (
        bmg1,
        bmg2,
        torch.stack(ef1s),
        torch.stack(ef2s),
        torch.stack(gfs),
        torch.stack(y1s),
        torch.stack(y2s),
    )


def collate_cached_batch(batch):
    """Collate function for CachedPairDataset, directly returns pre-built batch."""
    if len(batch) == 1:
        return batch[0]
    bmg1s, bmg2s, ef1s, ef2s, gfs, y1s, y2s = zip(*batch)
    return (
        bmg1s[0] if len(bmg1s) == 1 else bmg1s,
        bmg2s[0] if len(bmg2s) == 1 else bmg2s,
        ef1s[0] if len(ef1s) == 1 else torch.cat(ef1s),
        ef2s[0] if len(ef2s) == 1 else torch.cat(ef2s),
        gfs[0] if len(gfs) == 1 else torch.cat(gfs),
        y1s[0] if len(y1s) == 1 else torch.cat(y1s),
        y2s[0] if len(y2s) == 1 else torch.cat(y2s),
    )


def _collate_surrogate(batch):
    graphs, targets = zip(*batch)
    bmg = BatchMolGraph(list(graphs))
    return bmg, torch.stack(targets)
