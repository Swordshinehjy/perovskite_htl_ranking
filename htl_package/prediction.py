"""
Prediction logic for HTL Ranking Model.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer

from .constants import logger, DEVICE, GLOBAL_DIM, EXTRA_COLS, GLOBAL_COLS, TASK_NAMES, NUM_TASKS
from .configs import ModelConfig
from .features import _extra_feat, _extra_feat_single, _global_feat
from .datasets import ListDataset, collate_list_fn
from .models import HTLRankingModel
from .checkpoint import load_checkpoint, load_model_for_inference


@torch.no_grad()
def predict_pair(
    smiles_1: str,
    smiles_2: str,
    extra_raw_1: np.ndarray,
    extra_raw_2: np.ndarray,
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    global_feat: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Predict PCE ranking for a single pair of new structures.
    """
    model, scaler, config = load_model_for_inference(checkpoint_dir, checkpoint_name)

    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES format")

    atom_featurizer = MultiHotAtomFeaturizer.v2()
    bond_featurizer = MultiHotBondFeaturizer()
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
    )
    mg1 = BatchMolGraph([featurizer(mol1)])
    mg2 = BatchMolGraph([featurizer(mol2)])

    ef1 = torch.tensor(scaler.transform(extra_raw_1.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    ef2 = torch.tensor(scaler.transform(extra_raw_2.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)

    if global_feat is not None:
        gf = torch.tensor(global_feat.reshape(1, -1),
                          dtype=torch.float32).to(DEVICE)
    else:
        gf = torch.zeros(1, GLOBAL_DIM, dtype=torch.float32).to(DEVICE)

    s1, s2 = model(mg1, ef1, mg2, ef2, gf)
    s1 = s1.cpu().numpy()[0]
    s2 = s2.cpu().numpy()[0]

    scores_1 = {n: float(s1[i]) for i, n in enumerate(TASK_NAMES)}
    scores_2 = {n: float(s2[i]) for i, n in enumerate(TASK_NAMES)}
    ranking = {
        n: ("HTL_1" if s1[i] > s2[i] else "HTL_2")
        for i, n in enumerate(TASK_NAMES)
    }

    return {
        "scores_1": scores_1,
        "scores_2": scores_2,
        "ranking": ranking,
    }


@torch.no_grad()
def predict_batch(
    df_new: pd.DataFrame,
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch predict all new HTL material pairs in DataFrame.
    """
    model, scaler, config = load_model_for_inference(checkpoint_dir, checkpoint_name)

    df = df_new.copy().reset_index(drop=True)
    for s in ["1", "2"]:
        df[f"mol_{s}"] = df[f"SMILES_{s}"].apply(Chem.MolFromSmiles)

    valid = df.dropna(subset=["mol_1", "mol_2"]).copy()
    logger.info(f"Valid pairs: {len(valid)}/{len(df)}")

    ef1 = scaler.transform(_extra_feat(valid, "1"))
    ef2 = scaler.transform(_extra_feat(valid, "2"))

    if all(col in valid.columns for col in GLOBAL_COLS):
        gf = _global_feat(valid)
    else:
        gf = np.zeros((len(valid), GLOBAL_DIM), dtype=np.float32)

    atom_featurizer = MultiHotAtomFeaturizer.v2()
    bond_featurizer = MultiHotBondFeaturizer()
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
    )

    all_s1, all_s2 = [], []
    for i in range(len(valid)):
        mol1 = valid["mol_1"].iloc[i]
        mol2 = valid["mol_2"].iloc[i]
        mg1 = BatchMolGraph([featurizer(mol1)])
        mg2 = BatchMolGraph([featurizer(mol2)])
        t1 = torch.tensor(ef1[i:i + 1], dtype=torch.float32).to(DEVICE)
        t2 = torch.tensor(ef2[i:i + 1], dtype=torch.float32).to(DEVICE)
        gf_t = torch.tensor(gf[i:i + 1], dtype=torch.float32).to(DEVICE)
        s1, s2 = model(mg1, t1, mg2, t2, gf_t)
        all_s1.append(s1.cpu().numpy()[0])
        all_s2.append(s2.cpu().numpy()[0])

    S1 = np.array(all_s1)
    S2 = np.array(all_s2)

    for i, n in enumerate(TASK_NAMES):
        valid[f"score_{n}_1"] = S1[:, i]
        valid[f"score_{n}_2"] = S2[:, i]
        valid[f"preferred_{n}"] = np.where(S1[:, i] > S2[:, i],
                                           valid["Materials_1"],
                                           valid["Materials_2"])
    if output_path:
        valid.to_csv(output_path, index=False)
        logger.info(f"Predictions saved → {output_path}")
    return valid


@torch.no_grad()
def predict_list(
    df_list: pd.DataFrame,
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    output_path: Optional[str] = None,
    batch_size: int = 32,
    global_feat: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Perform ranking prediction on a single material list.
    """
    model, scaler, config = load_model_for_inference(checkpoint_dir, checkpoint_name)

    list_ds = ListDataset(df_list, scaler, global_feat=global_feat)
    list_loader = DataLoader(
        list_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_list_fn,
        num_workers=0,
    )

    logger.info(f"Valid materials: {len(list_ds)}/{len(df_list)}")

    all_scores = []
    all_materials = []

    for bmg, ef, gf, materials in list_loader:
        ef = ef.to(DEVICE)
        gf = gf.to(DEVICE)
        scores = model.encode(bmg, ef, gf)
        all_scores.append(scores.cpu().numpy())
        all_materials.extend(materials)

    scores = np.concatenate(all_scores, axis=0)

    result_df = list_ds.df.copy()
    result_df["Materials"] = list_ds.materials

    for i, name in enumerate(TASK_NAMES):
        result_df[f"score_{name}"] = scores[:, i]

    result_df = result_df.sort_values(by=f"score_{TASK_NAMES[0]}",
                                      ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)

    result_df = result_df[
        ["rank", "Materials"] + [f"score_{n}" for n in TASK_NAMES] + list(
            result_df.columns.difference(["rank", "Materials"] +
                                         [f"score_{n}" for n in TASK_NAMES]))]

    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"List ranking saved → {output_path}")

    return result_df
