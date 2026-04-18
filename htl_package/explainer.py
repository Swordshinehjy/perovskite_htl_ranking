"""Integrated Gradients attribution and differential attribution explainers."""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer

from .constants import EXTRA_COLS, EXTRA_DIM, GLOBAL_COLS, GLOBAL_DIM, TASK_NAMES, NUM_TASKS, DEVICE, logger
from .models import HTLRankingModel
from .checkpoint import load_checkpoint, load_model_for_inference
from .features import _extra_feat, _extra_feat_single, _global_feat
from .visualization import (
    attr_to_rgb,
    draw_mol_attribution,
    draw_feature_attribution,
    draw_score_ranking,
    merge_csvs,
    mol_to_image,
)

# IGExplainer


class IGExplainer:
    """
    Integrated Gradients (IG) explainer.

    Uses joint interpolation to satisfy IG completeness axiom:
    score(input) - score(baseline) = Σ(atom_IG) + Σ(extra_IG) + Σ(global_IG)
    """

    def __init__(
        self,
        model: HTLRankingModel,
        scaler: StandardScaler,
        n_steps: int = 100,
        target_task: int = 0,
    ):
        self.model = model
        self.scaler = scaler
        self.n_steps = n_steps
        self.target_task = target_task
        self.model.eval()

        self.featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=MultiHotAtomFeaturizer.v2(),
            bond_featurizer=MultiHotBondFeaturizer(),
        )

    def _warm_up(self, bmg: BatchMolGraph, ef: torch.Tensor, gf: torch.Tensor):
        with torch.no_grad():
            _ = self.model.encode(bmg, ef, gf)

    @torch.enable_grad()
    def _joint_ig(
        self,
        bmg: BatchMolGraph,
        ef: torch.Tensor,
        gf: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute IG for all inputs jointly using simultaneous interpolation.

        This satisfies the IG completeness axiom:
        score(input) - score(baseline) = Σ(atom_IG) + Σ(extra_IG) + Σ(global_IG)

        Returns:
            atom_ig: [n_atoms] atom-level IG
            extra_ig: [extra_dim] extra feature IG
            global_ig: [global_dim] global feature IG
        """
        self._warm_up(bmg, ef, gf)

        atom_orig = bmg.V.detach().clone()
        ef_orig = ef.detach().clone()
        gf_orig = gf.detach().clone()

        atom_baseline = torch.zeros_like(atom_orig)
        ef_baseline = torch.zeros_like(ef_orig)
        gf_baseline = torch.zeros_like(gf_orig)

        all_atom_grads: List[torch.Tensor] = []
        all_ef_grads: List[torch.Tensor] = []
        all_gf_grads: List[torch.Tensor] = []

        for alpha in np.linspace(0.0, 1.0, self.n_steps):
            atom_interp = (atom_baseline + alpha *
                           (atom_orig - atom_baseline)).clone()
            atom_interp.requires_grad_(True)

            ef_interp = (ef_baseline + alpha * (ef_orig - ef_baseline)).clone()
            ef_interp.requires_grad_(True)

            gf_interp = (gf_baseline + alpha * (gf_orig - gf_baseline)).clone()
            gf_interp.requires_grad_(True)

            bmg.V = atom_interp
            score = self.model.encode(bmg, ef_interp,
                                      gf_interp)[:, self.target_task]
            score.sum().backward()

            if atom_interp.grad is not None:
                all_atom_grads.append(atom_interp.grad.detach().clone())
            if ef_interp.grad is not None:
                all_ef_grads.append(ef_interp.grad.detach().clone())
            if gf_interp.grad is not None:
                all_gf_grads.append(gf_interp.grad.detach().clone())

        bmg.V = atom_orig

        if not all_atom_grads:
            n_atoms = atom_orig.shape[0]
            return (np.zeros(n_atoms), np.zeros(ef_orig.shape[-1]),
                    np.zeros(gf_orig.shape[-1]))

        avg_atom_grad = torch.stack(all_atom_grads).mean(dim=0)
        atom_ig = (atom_orig - atom_baseline) * avg_atom_grad
        atom_ig = atom_ig.sum(dim=-1).cpu().numpy()

        avg_ef_grad = torch.stack(all_ef_grads).mean(dim=0)
        ef_ig = (ef_orig - ef_baseline) * avg_ef_grad
        ef_ig = ef_ig.cpu().numpy()[0]

        avg_gf_grad = torch.stack(all_gf_grads).mean(dim=0)
        gf_ig = (gf_orig - gf_baseline) * avg_gf_grad
        gf_ig = gf_ig.cpu().numpy()[0]

        return atom_ig, ef_ig, gf_ig

    @torch.enable_grad()
    def _atom_ig(
        self,
        bmg: BatchMolGraph,
        ef: torch.Tensor,
        gf: torch.Tensor,
    ) -> np.ndarray:
        self._warm_up(bmg, ef, gf)

        atom_feats_orig = bmg.V.detach().clone()
        baseline = torch.zeros_like(atom_feats_orig)

        all_grads: List[torch.Tensor] = []

        for alpha in np.linspace(0.0, 1.0, self.n_steps):
            interp = (baseline + alpha * (atom_feats_orig - baseline)).clone()
            interp.requires_grad_(True)

            bmg.V = interp

            score = self.model.encode(bmg, ef, gf)[:, self.target_task]
            score.sum().backward()

            if interp.grad is not None:
                all_grads.append(interp.grad.detach().clone())

        bmg.V = atom_feats_orig

        if not all_grads:
            return np.zeros(atom_feats_orig.shape[0])

        avg_grad = torch.stack(all_grads).mean(dim=0)
        ig = (atom_feats_orig - baseline) * avg_grad
        return ig.sum(dim=-1).cpu().numpy()

    @torch.enable_grad()
    def _extra_ig(
        self,
        bmg: BatchMolGraph,
        ef: torch.Tensor,
        gf: torch.Tensor,
    ) -> np.ndarray:
        ef_orig = ef.detach().clone()
        baseline = torch.zeros_like(ef_orig)

        all_grads: List[torch.Tensor] = []

        for alpha in np.linspace(0.0, 1.0, self.n_steps):
            interp = (baseline + alpha * (ef_orig - baseline)).clone()
            interp.requires_grad_(True)

            score = self.model.encode(bmg, interp, gf)[:, self.target_task]
            score.sum().backward()

            if interp.grad is not None:
                all_grads.append(interp.grad.detach().clone())

        if not all_grads:
            return np.zeros(ef_orig.shape[-1])

        avg_grad = torch.stack(all_grads).mean(dim=0)
        ig = (ef_orig - baseline) * avg_grad
        return ig.cpu().numpy()[0]

    def explain_molecule(
        self,
        smiles: str,
        ef_scaled: np.ndarray,
        gf: np.ndarray,
        material_name: str = "material",
        save_dir: str = "explain_output",
        extra_raw: Optional[np.ndarray] = None,
        global_feat: Optional[np.ndarray] = None,
        use_joint_ig: bool = True,
    ) -> Dict[str, Any]:
        if ef_scaled is None:
            if extra_raw is None:
                raise ValueError("Must provide ef_scaled or extra_raw")
            ef_scaled = self.scaler.transform(extra_raw.reshape(1, -1))[0]
        if gf is None:
            gf = global_feat if global_feat is not None else np.zeros(
                GLOBAL_DIM, dtype=np.float32)

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        AllChem.Compute2DCoords(mol)

        bmg_score = BatchMolGraph([self.featurizer(mol)])
        bmg_ig = BatchMolGraph([self.featurizer(mol)])

        ef = torch.tensor(ef_scaled.reshape(1, -1),
                          dtype=torch.float32).to(DEVICE)
        gf_t = torch.tensor(gf.reshape(1, -1), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            score = self.model.encode(bmg_score, ef,
                                      gf_t)[0, self.target_task].item()

        logger.info(
            f"[{material_name}] Predicted score ({TASK_NAMES[self.target_task]}): {score:.4f}"
        )

        n_heavy = mol.GetNumAtoms()

        if use_joint_ig:
            logger.info(
                f"[{material_name}] Computing joint IG ({self.n_steps} steps)..."
            )
            atom_attrs, extra_attrs, global_attrs = self._joint_ig(
                bmg_ig, ef, gf_t)
            atom_attrs = atom_attrs[:n_heavy]
        else:
            logger.info(
                f"[{material_name}] Computing atom IG ({self.n_steps} steps)..."
            )
            atom_attrs = self._atom_ig(bmg_ig, ef, gf_t)
            atom_attrs = atom_attrs[:n_heavy]

            logger.info(
                f"[{material_name}] Computing extra-feature IG ({self.n_steps} steps)..."
            )
            bmg_ef_ig = BatchMolGraph([self.featurizer(mol)])
            extra_attrs = self._extra_ig(bmg_ef_ig, ef, gf_t)
            global_attrs = np.zeros(GLOBAL_DIM, dtype=np.float32)

        extra_names = [c.replace("_{s}", "") for c in EXTRA_COLS]
        pos_atoms = [i for i, v in enumerate(atom_attrs) if v > 0]

        extra_to_atom = None
        combined_atom_attrs = atom_attrs.copy()

        safe_name = material_name.replace("/", "_").replace(" ", "_")
        png_path = str(Path(save_dir) / f"{safe_name}_mol.png")
        bar_path = str(Path(save_dir) / f"{safe_name}_features.png")
        csv_path = str(Path(save_dir) / f"{safe_name}_summary.csv")

        draw_mol_attribution(
            mol,
            combined_atom_attrs,
            title=material_name,
            score=score,
            save_path=png_path,
            task_name=TASK_NAMES[self.target_task],
        )
        draw_feature_attribution(
            extra_attrs,
            extra_names,
            save_path=bar_path,
            title=material_name,
            score=score,
            task_name=TASK_NAMES[self.target_task],
        )
        self._save_summary_csv(
            material_name,
            smiles,
            score,
            atom_attrs,
            extra_attrs,
            extra_names,
            csv_path,
            extra_to_atom=extra_to_atom,
            combined_atom_attrs=combined_atom_attrs,
        )

        return {
            "score": score,
            "atom_attrs": atom_attrs,
            "extra_to_atom": extra_to_atom,
            "combined_atom_attrs": combined_atom_attrs,
            "extra_attrs": extra_attrs,
            "global_attrs": global_attrs,
            "extra_names": extra_names,
            "pos_atoms": pos_atoms,
            "mol_png_path": png_path,
            "bar_chart_path": bar_path,
            "summary_csv_path": csv_path,
        }

    @staticmethod
    def _save_summary_csv(
        material_name: str,
        smiles: str,
        score: float,
        atom_attrs: np.ndarray,
        extra_attrs: np.ndarray,
        extra_names: list,
        csv_path: str,
        extra_to_atom: Optional[np.ndarray] = None,
        combined_atom_attrs: Optional[np.ndarray] = None,
    ):
        rows = []
        has_extra_to_atom = extra_to_atom is not None and combined_atom_attrs is not None
        for i, v in enumerate(atom_attrs):
            row = {
                "material":
                material_name,
                "type":
                "atom",
                "index":
                i,
                "name":
                f"atom_{i}",
                "attribution_mpnn":
                float(v),
                "attribution":
                float(combined_atom_attrs[i])
                if has_extra_to_atom else float(v),
                "sign":
                "positive" if
                (combined_atom_attrs[i] > 0 if has_extra_to_atom else v > 0)
                else
                ("negative" if
                 (combined_atom_attrs[i] < 0 if has_extra_to_atom else v < 0)
                 else "zero"),
            }
            if has_extra_to_atom:
                row["attribution_extra"] = float(extra_to_atom[i])
            rows.append(row)
        for i, (n, v) in enumerate(zip(extra_names, extra_attrs)):
            rows.append({
                "material":
                material_name,
                "type":
                "extra_feature",
                "index":
                i,
                "name":
                n,
                "attribution":
                float(v),
                "sign":
                "positive" if v > 0 else ("negative" if v < 0 else "zero"),
            })

        df_out = pd.DataFrame(rows)
        df_out["smiles"] = smiles
        df_out["predicted_score"] = score
        df_out["task"] = TASK_NAMES[0]
        df_out.to_csv(csv_path, index=False)
        logger.info(f"  Summary CSV → {csv_path}")


# DiffAttrExplainer


class DiffAttrExplainer:
    """
    Differential Attribution Explainer for molecular pairs.

    For a pair (mol_1, mol_2) the model has already scored, we answer:
      "What structural and feature differences drive the predicted preference?"

    Two complementary analyses are computed:

    1. atom-level structural diff
       IG attributions are computed independently for each molecule
       (baseline = zero atom-feature vector, as in IGExplainer).
       A Maximum Common Substructure (MCS) search then partitions each
       molecule into:
         - MCS scaffold atoms  "shared" structural contribution
         - Unique atoms - "differential" structural contribution
       The unique-atom attribution sums for mol_1 and mol_2 directly
       quantify how much the *structural difference* benefits each molecule.

    2. extra-feature differential IG
       Because both molecules share the same EXTRA_DIM feature space,
       we define:
           Δattr[i] = IG_2[i] − IG_1[i]
       Δattr[i] > 0  : feature i favors mol_2 (red)
       Δattr[i] < 0  : feature i favors mol_1 (blue)
    """

    def __init__(
        self,
        model: HTLRankingModel,
        scaler: StandardScaler,
        n_steps: int = 100,
        target_task: int = 0,
    ):
        self.model = model
        self.scaler = scaler
        self.n_steps = n_steps
        self.target_task = target_task
        self.model.eval()

        self.featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=MultiHotAtomFeaturizer.v2(),
            bond_featurizer=MultiHotBondFeaturizer(),
        )
        self._ig = IGExplainer(model=model,
                               scaler=scaler,
                               n_steps=n_steps,
                               target_task=target_task)

    def _score_and_attrs(
        self,
        smiles: str,
        ef_scaled: np.ndarray,
        gf: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute score + atom IG + extra IG for one molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        AllChem.Compute2DCoords(mol)

        ef = torch.tensor(ef_scaled.reshape(1, -1),
                          dtype=torch.float32).to(DEVICE)
        gf_t = torch.tensor(gf.reshape(1, -1), dtype=torch.float32).to(DEVICE)

        # score
        bmg_s = BatchMolGraph([self.featurizer(mol)])
        with torch.no_grad():
            score = self.model.encode(bmg_s, ef,
                                      gf_t)[0, self.target_task].item()

        # joint IG (consistent with IGExplainer.explain_molecule default)
        bmg_ig = BatchMolGraph([self.featurizer(mol)])
        atom_attrs, extra_attrs, _global_attrs = self._ig._joint_ig(
            bmg_ig, ef, gf_t)
        atom_attrs = atom_attrs[:mol.GetNumAtoms()]

        return dict(mol=mol,
                    score=score,
                    atom_attrs=atom_attrs,
                    extra_attrs=extra_attrs)

    @staticmethod
    def _find_mcs(mol1, mol2) -> Dict[str, Any]:
        """
        Find Maximum Common Substructure between mol1 and mol2.

        Returns a dict with:
          mcs_smarts       : SMARTS of the MCS pattern (None if no MCS found)
          mcs_num_atoms    : number of atoms in MCS
          mol1_mcs_atoms   : set of atom indices in mol1 matching MCS
          mol2_mcs_atoms   : set of atom indices in mol2 matching MCS
          mol1_unique_atoms: atoms in mol1 NOT in MCS  (structural diff)
          mol2_unique_atoms: atoms in mol2 NOT in MCS  (structural diff)
        """
        try:
            result = rdFMCS.FindMCS(
                [mol1, mol2],
                timeout=30,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder,
                completeRingsOnly=False,
                ringMatchesRingOnly=False,
            )
            if result.numAtoms == 0:
                raise ValueError("MCS is empty")

            mcs_mol = Chem.MolFromSmarts(result.smartsString)
            match1 = set(mol1.GetSubstructMatch(mcs_mol))
            match2 = set(mol2.GetSubstructMatch(mcs_mol))

            return dict(
                mcs_smarts=result.smartsString,
                mcs_num_atoms=result.numAtoms,
                mol1_mcs_atoms=match1,
                mol2_mcs_atoms=match2,
                mol1_unique_atoms=set(range(mol1.GetNumAtoms())) - match1,
                mol2_unique_atoms=set(range(mol2.GetNumAtoms())) - match2,
            )
        except Exception as exc:
            logger.warning(
                f"MCS search failed ({exc}); treating all atoms as unique.")
            return dict(
                mcs_smarts=None,
                mcs_num_atoms=0,
                mol1_mcs_atoms=set(),
                mol2_mcs_atoms=set(),
                mol1_unique_atoms=set(range(mol1.GetNumAtoms())),
                mol2_unique_atoms=set(range(mol2.GetNumAtoms())),
            )

    @staticmethod
    def _struct_summary(atom_attrs_1, atom_attrs_2,
                        mcs_info) -> Dict[str, float]:
        """Aggregate atom attributions into MCS-scaffold vs unique-fragment buckets."""

        def _safe_sum(attrs, idx_set):
            idx = list(idx_set)
            return float(attrs[idx].sum()) if idx else 0.0

        return dict(
            mcs_attr_1=_safe_sum(atom_attrs_1, mcs_info["mol1_mcs_atoms"]),
            mcs_attr_2=_safe_sum(atom_attrs_2, mcs_info["mol2_mcs_atoms"]),
            unique_attr_1=_safe_sum(atom_attrs_1,
                                    mcs_info["mol1_unique_atoms"]),
            unique_attr_2=_safe_sum(atom_attrs_2,
                                    mcs_info["mol2_unique_atoms"]),
            mcs_num_atoms=mcs_info["mcs_num_atoms"],
            unique_num_atoms_1=len(mcs_info["mol1_unique_atoms"]),
            unique_num_atoms_2=len(mcs_info["mol2_unique_atoms"]),
        )

    def explain_pair(
        self,
        smiles_1: str,
        smiles_2: str,
        ef_scaled_1: np.ndarray,
        ef_scaled_2: np.ndarray,
        gf: np.ndarray,
        name_1: str = "HTL_1",
        name_2: str = "HTL_2",
        pair_id: str = "pair_0",
        save_dir: str = "diff_output",
    ) -> Dict[str, Any]:
        """
        Full differential attribution for one (smiles_1, smiles_2) pair.

        Parameters
        ----------
        smiles_1/2    : SMILES strings
        ef_scaled_1/2 : StandardScaler-transformed extra feature vectors [EXTRA_DIM]
        gf            : Global feature vector [GLOBAL_DIM] (same for both mols in a pair)
        name_1/2      : Display names
        pair_id       : File-safe identifier used as filename prefix
        save_dir      : Output directory

        Returns
        -------
        dict with all attribution arrays and output file paths
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"  [{pair_id}] Attributions for {name_1} ...")
        r1 = self._score_and_attrs(smiles_1, ef_scaled_1, gf)

        logger.info(f"  [{pair_id}] Attributions for {name_2} ...")
        r2 = self._score_and_attrs(smiles_2, ef_scaled_2, gf)

        # differential extra-feature attribution
        # Δattr[i] = IG_2[i] − IG_1[i]
        # interpretation: positive → feature i favors mol_2 in this pair
        delta_extra = r2["extra_attrs"] - r1["extra_attrs"]
        extra_names = [c.replace("_{s}", "") for c in EXTRA_COLS]

        # MCS structural analysis
        logger.info(f"  [{pair_id}] Computing MCS ...")
        mcs_info = self._find_mcs(r1["mol"], r2["mol"])
        struct_info = self._struct_summary(r1["atom_attrs"], r2["atom_attrs"],
                                           mcs_info)

        safe = pair_id.replace("/", "_").replace(" ", "_")
        cmp_path = str(Path(save_dir) / f"{safe}_comparison.png")
        feat_path = str(Path(save_dir) / f"{safe}_diff_features.png")
        csv_path = str(Path(save_dir) / f"{safe}_diff_summary.csv")

        # Draw
        self._draw_pair_comparison(
            r1["mol"],
            r2["mol"],
            r1["atom_attrs"],
            r2["atom_attrs"],
            mcs_info,
            struct_info,
            name_1,
            name_2,
            r1["score"],
            r2["score"],
            cmp_path,
        )
        self._draw_diff_features(
            extra_names,
            r1["extra_attrs"],
            r2["extra_attrs"],
            delta_extra,
            name_1,
            name_2,
            r1["score"],
            r2["score"],
            feat_path,
        )
        self._save_diff_csv(
            pair_id,
            name_1,
            name_2,
            smiles_1,
            smiles_2,
            r1["score"],
            r2["score"],
            r1["atom_attrs"],
            r2["atom_attrs"],
            r1["extra_attrs"],
            r2["extra_attrs"],
            delta_extra,
            extra_names,
            mcs_info,
            struct_info,
            csv_path,
        )

        return dict(
            pair_id=pair_id,
            name_1=name_1,
            name_2=name_2,
            score_1=r1["score"],
            score_2=r2["score"],
            score_diff=r2["score"] - r1["score"],
            preferred=name_1 if r1["score"] >= r2["score"] else name_2,
            atom_attrs_1=r1["atom_attrs"],
            atom_attrs_2=r2["atom_attrs"],
            extra_attrs_1=r1["extra_attrs"],
            extra_attrs_2=r2["extra_attrs"],
            delta_extra=delta_extra,
            extra_names=extra_names,
            mcs_info=mcs_info,
            struct_info=struct_info,
            comparison_path=cmp_path,
            diff_feat_path=feat_path,
            csv_path=csv_path,
        )

    def _draw_pair_comparison(
        self,
        mol1,
        mol2,
        atom_attrs_1: np.ndarray,
        atom_attrs_2: np.ndarray,
        mcs_info: Dict,
        struct_info: Dict,
        name_1: str,
        name_2: str,
        score_1: float,
        score_2: float,
        save_path: str,
    ):
        """
        2×2 panel figure:
          [top-left]  mol_1 with atom attribution heatmap
          [top-right] mol_2 with atom attribution heatmap
          [bot-left]  score comparison bar chart
          [bot-right] MCS scaffold vs unique-fragment attribution breakdown
        Unique atoms (not in MCS) are annotated with ★ in the title lists.
        """
        winner = name_1 if score_1 >= score_2 else name_2
        score_diff = abs(score_1 - score_2)

        fig = plt.figure(figsize=(17, 13))
        gs = gridspec.GridSpec(2,
                               2,
                               figure=fig,
                               hspace=0.40,
                               wspace=0.28,
                               left=0.05,
                               right=0.88,
                               top=0.90,
                               bottom=0.06)

        ax_m1 = fig.add_subplot(gs[0, 0])
        ax_m2 = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[1, 0])
        ax_str = fig.add_subplot(gs[1, 1])

        # molecule panels
        for ax, mol, attrs, name, score, mcs_idx, uniq_idx in [
            (ax_m1, mol1, atom_attrs_1, name_1, score_1,
             mcs_info["mol1_mcs_atoms"], mcs_info["mol1_unique_atoms"]),
            (ax_m2, mol2, atom_attrs_2, name_2, score_2,
             mcs_info["mol2_mcs_atoms"], mcs_info["mol2_unique_atoms"]),
        ]:
            img = mol_to_image(mol, attrs)
            ax.imshow(img)
            ax.axis("off")
            # annotate unique (non-MCS) atoms
            uniq_sorted = sorted(uniq_idx)
            uniq_str = (f"★ unique: {uniq_sorted}"
                        if uniq_sorted else "no unique atoms vs MCS")
            title_color = "#d62728" if name == winner else "#333333"
            title_prefix = "▶ " if name == winner else "  "
            ax.set_title(
                f"{title_prefix}{name}   Score: {score:.4f}\n{uniq_str}",
                fontsize=10,
                fontweight="bold",
                color=title_color,
            )
            # top-5 atom attributions as text box
            top5 = np.argsort(np.abs(attrs))[-5:][::-1]
            lines = [
                f"atom {i}: {attrs[i]:+.3f}{'★' if i in uniq_idx else ''}"
                for i in top5
            ]
            ax.text(0.02,
                    0.02,
                    "\n".join(lines),
                    fontsize=7.5,
                    transform=ax.transAxes,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white",
                              alpha=0.80))

        # score bar
        bar_colors = [
            "#d62728" if name_1 == winner else "#aaaaaa",
            "#d62728" if name_2 == winner else "#aaaaaa"
        ]
        bars = ax_bar.bar([name_1, name_2], [score_1, score_2],
                          color=bar_colors,
                          edgecolor="white",
                          linewidth=0.8,
                          width=0.45)

        score_min = min(score_1, score_2)
        score_max = max(score_1, score_2)
        score_range = score_max - score_min if score_max != score_min else 1.0

        for bar, v in zip(bars, [score_1, score_2]):
            offset = score_range * 0.05
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        v + (offset if v >= 0 else -offset),
                        f"{v:.4f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=11,
                        fontweight="bold")

        ax_bar.set_ylabel(f"Predicted {TASK_NAMES[self.target_task]} Score",
                          fontsize=10)
        ax_bar.set_title("Score Comparison", fontsize=11, fontweight="bold")

        if score_min >= 0:
            ax_bar.set_ylim(0, score_max * 1.18)
        else:
            margin = score_range * 0.18
            ax_bar.set_ylim(score_min - margin, score_max + margin)

        ax_bar.axhline(0,
                       color="black",
                       linewidth=0.5,
                       linestyle="-",
                       alpha=0.3)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.tick_params(axis="x", labelsize=10)

        # structural breakdown
        mcs_n = struct_info["mcs_num_atoms"]
        un1 = struct_info["unique_num_atoms_1"]
        un2 = struct_info["unique_num_atoms_2"]
        labels = [
            f"MCS scaffold\n({mcs_n} atoms)",
            f"Unique fragment\n({un1} / {un2} atoms)",
        ]
        vals_1 = [struct_info["mcs_attr_1"], struct_info["unique_attr_1"]]
        vals_2 = [struct_info["mcs_attr_2"], struct_info["unique_attr_2"]]

        x = np.arange(len(labels))
        width = 0.30
        b1 = ax_str.bar(x - width / 2,
                        vals_1,
                        width,
                        label=name_1,
                        color="#d62728",
                        alpha=0.82)
        b2 = ax_str.bar(x + width / 2,
                        vals_2,
                        width,
                        label=name_2,
                        color="#1f77b4",
                        alpha=0.82)
        ax_str.set_xticks(x)
        ax_str.set_xticklabels(labels, fontsize=9)
        ax_str.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_str.set_ylabel("Σ Atom Attribution", fontsize=10)
        ax_str.set_title(
            f"Structural Attribution Breakdown\n"
            f"MCS: {mcs_n} atoms  |  ★ unique diff visualised above",
            fontsize=10,
            fontweight="bold",
        )
        ax_str.legend(fontsize=8)
        ax_str.spines["top"].set_visible(False)
        ax_str.spines["right"].set_visible(False)
        # value labels on structural bars
        for bar, v in [(bb, vv) for bset, vvals in [(b1, vals_1), (b2, vals_2)]
                       for bb, vv in zip(bset, vvals)]:
            if abs(v) > 1e-5:
                ax_str.text(bar.get_x() + bar.get_width() / 2,
                            v + np.sign(v) * 0.005 *
                            (abs(vals_1[0]) + abs(vals_2[0]) + 1e-6),
                            f"{v:+.3f}",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=8)

        # shared colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "attr", ["#1f77b4", "white", "#d62728"])
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.90, 0.55, 0.015, 0.32])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Normalised Atom Attribution", fontsize=9)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["−1\n(neg)", "0", "+1\n(pos)"])

        fig.suptitle(
            f"Differential Attribution: {name_1}  vs  {name_2}\n"
            f"Preferred: ▶ {winner}    |    Δscore = {score_diff:.4f}",
            fontsize=13,
            fontweight="bold",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Comparison figure → {save_path}")

    @staticmethod
    def _draw_diff_features(
        extra_names: List[str],
        attrs_1: np.ndarray,
        attrs_2: np.ndarray,
        delta: np.ndarray,
        name_1: str,
        name_2: str,
        score_1: float,
        score_2: float,
        save_path: str,
    ):
        """
        3-panel horizontal figure:
          Panel A – individual IG for mol_1 (sorted by value)
          Panel B – individual IG for mol_2 (sorted by value)
          Panel C – Differential IG: Δattr = IG_2 − IG_1
                    red  → feature favors mol_2
                    blue → feature favors mol_1
        """
        n = len(extra_names)
        winner = name_1 if score_1 >= score_2 else name_2
        fig, axes = plt.subplots(1,
                                 3,
                                 figsize=(19, max(5, n * 0.52)),
                                 constrained_layout=True)

        def _panel(ax, attrs, name, score, is_winner):
            order = np.argsort(attrs)
            sa, sn = attrs[order], [extra_names[i] for i in order]
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in sa]
            ax.barh(range(n),
                    sa,
                    color=colors,
                    edgecolor="white",
                    linewidth=0.4)
            ax.set_yticks(range(n))
            ax.set_yticklabels(sn, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("IG Attribution", fontsize=10)
            title_color = "#d62728" if is_winner else "#444444"
            prefix = "▶ " if is_winner else "   "
            ax.set_title(f"{prefix}{name}\nScore: {score:.4f}",
                         fontsize=10,
                         fontweight="bold",
                         color=title_color)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        _panel(axes[0], attrs_1, name_1, score_1, name_1 == winner)
        _panel(axes[1], attrs_2, name_2, score_2, name_2 == winner)

        # differential panel
        order_d = np.argsort(delta)
        sd, snd = delta[order_d], [extra_names[i] for i in order_d]
        colors_d = ["#d62728" if v > 0 else "#1f77b4" for v in sd]
        bars = axes[2].barh(range(n),
                            sd,
                            color=colors_d,
                            edgecolor="white",
                            linewidth=0.4)
        axes[2].set_yticks(range(n))
        axes[2].set_yticklabels(snd, fontsize=9)
        axes[2].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[2].set_xlabel("Δ IG  (IG₂ − IG₁)", fontsize=10)
        axes[2].set_title(
            f"Differential Attribution\n"
            f"← {name_1}  |  → {name_2}",
            fontsize=10,
            fontweight="bold",
        )
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)

        x_max = np.abs(sd).max() if len(sd) > 0 else 1.0
        offset = x_max * 0.015
        axes[2].set_xlim(-x_max * 1.30, x_max * 1.30)
        for bar, val in zip(bars, sd):
            if abs(val) < 1e-8:
                continue
            axes[2].text(
                val + (offset if val >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=7.5,
                color="black",
            )

        pos_p = mpatches.Patch(color="#d62728", label=f"Favors {name_2}")
        neg_p = mpatches.Patch(color="#1f77b4", label=f"Favors {name_1}")
        axes[2].legend(handles=[pos_p, neg_p], loc="lower right", fontsize=8)

        fig.suptitle(
            f"Extra-Feature Attribution: {name_1} vs {name_2}",
            fontsize=12,
            fontweight="bold",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Differential feature chart → {save_path}")

    @staticmethod
    def _save_diff_csv(
        pair_id,
        name_1,
        name_2,
        smiles_1,
        smiles_2,
        score_1,
        score_2,
        atom_attrs_1,
        atom_attrs_2,
        extra_attrs_1,
        extra_attrs_2,
        delta_extra,
        extra_names,
        mcs_info,
        struct_info,
        csv_path,
    ):
        """Write all per-atom and per-feature attributions to a tidy CSV."""
        preferred = name_1 if score_1 >= score_2 else name_2
        score_diff = score_2 - score_1
        rows = []

        # extra-feature rows
        for i, fname in enumerate(extra_names):
            rows.append(
                dict(
                    pair_id=pair_id,
                    type="extra_feature",
                    name=fname,
                    molecule="both",
                    attr_1=float(extra_attrs_1[i]),
                    attr_2=float(extra_attrs_2[i]),
                    delta_attr=float(delta_extra[i]),
                    favors=(name_2 if delta_extra[i] > 0 else
                            (name_1 if delta_extra[i] < 0 else "neutral")),
                    in_mcs=None,
                    is_unique=None,
                ))

        # atom rows for mol_1
        for i, v in enumerate(atom_attrs_1):
            in_mcs = i in mcs_info["mol1_mcs_atoms"]
            rows.append(
                dict(
                    pair_id=pair_id,
                    type="atom",
                    name=f"atom_{i}",
                    molecule=name_1,
                    attr_1=float(v),
                    attr_2=None,
                    delta_attr=None,
                    favors=None,
                    in_mcs=in_mcs,
                    is_unique=not in_mcs,
                ))

        # or mol_2
        for i, v in enumerate(atom_attrs_2):
            in_mcs = i in mcs_info["mol2_mcs_atoms"]
            rows.append(
                dict(
                    pair_id=pair_id,
                    type="atom",
                    name=f"atom_{i}",
                    molecule=name_2,
                    attr_1=None,
                    attr_2=float(v),
                    delta_attr=None,
                    favors=None,
                    in_mcs=in_mcs,
                    is_unique=not in_mcs,
                ))

        df_out = pd.DataFrame(rows)
        df_out["smiles_1"] = smiles_1
        df_out["smiles_2"] = smiles_2
        df_out["score_1"] = score_1
        df_out["score_2"] = score_2
        df_out["score_diff"] = score_diff
        df_out["preferred"] = preferred
        df_out["mcs_num_atoms"] = mcs_info["mcs_num_atoms"]
        df_out["unique_num_atoms_1"] = struct_info["unique_num_atoms_1"]
        df_out["unique_num_atoms_2"] = struct_info["unique_num_atoms_2"]
        df_out["task"] = TASK_NAMES[0]

        df_out.to_csv(csv_path, index=False)
        logger.info(f"  Differential summary CSV → {csv_path}")


# Convenience functions


@torch.no_grad()
def explain(
    df_list: pd.DataFrame,
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    save_dir: str = "explain_output",
    n_steps: int = 50,
    batch_size: int = 1,
    global_feat: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Single-molecule IG attribution for a DataFrame of materials.

    Parameters
    ----------
    df_list : pd.DataFrame
        Must contain a SMILES column and extra feature columns.
    checkpoint_dir : str
    checkpoint_name : str
    save_dir : str
    n_steps : int
    batch_size : int  (unused, kept for API compatibility)
    global_feat : optional np.ndarray

    Returns
    -------
    List of result dicts (one per material).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model, scaler, model_cfg = load_model_for_inference(
        checkpoint_dir, checkpoint_name)

    explainer = IGExplainer(model=model,
                            scaler=scaler,
                            n_steps=n_steps,
                            target_task=0)

    smiles_col = next(
        (c for c in ["SMILES", "smiles", "Smiles"] if c in df_list.columns),
        None)
    if smiles_col is None:
        raise ValueError("DataFrame must contain SMILES column")

    mat_col = next(
        (c for c in
         ["Materials", "materials", "Material", "material", "Name", "name"]
         if c in df_list.columns), None)

    ef_all = _extra_feat_single(df_list)
    ef_all_scaled = scaler.transform(ef_all)

    if global_feat is not None:
        gf_all = np.tile(global_feat, (len(df_list), 1)).astype(np.float32)
    elif all(col in df_list.columns for col in GLOBAL_COLS):
        gf_all = _global_feat(df_list)
    else:
        gf_all = np.zeros((len(df_list), GLOBAL_DIM), dtype=np.float32)

    df_list = df_list.reset_index(drop=True)
    results = []
    all_scores = []

    for idx, row in df_list.iterrows():
        smiles = row[smiles_col]
        material_name = str(row[mat_col]) if mat_col else f"material_{idx}"
        ef_row_scaled = ef_all_scaled[idx]
        gf_row = gf_all[idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Explaining [{idx+1}/{len(df_list)}]: {material_name}")
        logger.info(f"{'='*60}")

        try:
            result = explainer.explain_molecule(
                smiles=smiles,
                ef_scaled=ef_row_scaled,
                gf=gf_row,
                material_name=material_name,
                save_dir=save_dir,
            )
            result["material_name"] = material_name
            result["smiles"] = smiles
            results.append(result)
            all_scores.append((material_name, result["score"]))
        except Exception as e:
            logger.error(f"  Failed to explain {material_name}: {e}")
            continue

    if len(all_scores) > 1:
        draw_score_ranking(all_scores, save_dir)

    merge_csvs(save_dir, "*_summary.csv", "all_attributions.csv")
    logger.info(f"\nExplain mode done. All outputs → {save_dir}/")
    return results


def diff_attr(
    df_pairs: pd.DataFrame,
    checkpoint_dir: str,
    checkpoint_name: str = "best_model",
    save_dir: str = "diff_output",
    n_steps: int = 50,
    global_feat: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Differential attribution analysis for a CSV of molecular pairs.

    Input CSV format (same as htl-new.csv / predict mode):
      Required  : SMILES_1, SMILES_2
                  extra-feature columns with _1 / _2 suffixes
                  Materials_1, Materials_2  (used as display names)
      Optional  : MO_ITO  (global feature)

    For each pair the function computes:
      • Atom-level IG for mol_1 and mol_2 with MCS-based decomposition
      • Differential extra-feature attribution  Δattr = IG_2 − IG_1

    Outputs written to save_dir/
      {pair_id}_comparison.png   — structural comparison heatmaps + breakdown
      {pair_id}_diff_features.png— feature-level differential attribution
      {pair_id}_diff_summary.csv — tidy per-atom / per-feature CSV
      all_diff_attributions.csv  — merged CSV across all pairs

    Returns
    -------
    List of result dicts (one per valid pair).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {checkpoint_dir}/{checkpoint_name}")
    model, scaler, model_cfg = load_model_for_inference(
        checkpoint_dir, checkpoint_name)

    explainer = DiffAttrExplainer(model=model, scaler=scaler, n_steps=n_steps)

    df = df_pairs.copy().reset_index(drop=True)
    for s in ["1", "2"]:
        df[f"mol_{s}"] = df[f"SMILES_{s}"].apply(Chem.MolFromSmiles)

    ef1_all = scaler.transform(_extra_feat(df, "1"))
    ef2_all = scaler.transform(_extra_feat(df, "2"))

    if global_feat is not None:
        gf_all = np.tile(global_feat, (len(df), 1)).astype(np.float32)
    elif all(col in df.columns for col in GLOBAL_COLS):
        gf_all = _global_feat(df)
    else:
        gf_all = np.zeros((len(df), GLOBAL_DIM), dtype=np.float32)

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        if row.get("mol_1") is None or row.get("mol_2") is None:
            logger.warning(f"Skipping row {idx}: invalid SMILES, skipping.")
            continue

        name_1 = str(row.get("Materials_1", f"HTL1_{idx}"))
        name_2 = str(row.get("Materials_2", f"HTL2_{idx}"))
        # short pair identifier that works as a filename prefix
        pair_id = f"pair{idx:03d}_{name_1[:20]}_vs_{name_2[:20]}"

        logger.info(f"\n{'='*64}")
        logger.info(f"Differential attribution [{idx+1}/{len(df)}]: {pair_id}")
        logger.info(f"{'='*64}")

        try:
            result = explainer.explain_pair(
                smiles_1=row["SMILES_1"],
                smiles_2=row["SMILES_2"],
                ef_scaled_1=ef1_all[idx],
                ef_scaled_2=ef2_all[idx],
                gf=gf_all[idx],
                name_1=name_1,
                name_2=name_2,
                pair_id=pair_id,
                save_dir=save_dir,
            )
            results.append(result)
        except Exception as exc:
            import traceback
            logger.error(f"  Failed for {pair_id}: {exc}")
            traceback.print_exc()
            continue

    merge_csvs(save_dir, "*_diff_summary.csv", "all_diff_attributions.csv")
    logger.info(
        f"\nDiff attribution complete. {len(results)} pairs → {save_dir}/")
    return results
