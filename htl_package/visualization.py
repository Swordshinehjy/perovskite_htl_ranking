"""Shared visualization utilities for attribution, molecule rendering, and score ranking."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import io
from PIL import Image
from cairosvg import svg2png as _svg2png
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from .constants import TASK_NAMES, logger

# Core color mapping


def attr_to_rgb(v: float) -> tuple:
    """Map a normalized attribution value in [-1, 1] to an RGB color.

    Negative → blue (#1f77b4), zero → white, positive → red (#d62728).
    """
    v = float(np.clip(v, -1.0, 1.0))
    blue = (0.122, 0.467, 0.706)
    red = (0.839, 0.153, 0.157)
    if v < 0:
        t = v + 1
        return (blue[0] + t * (1 - blue[0]), blue[1] + t * (1 - blue[1]),
                blue[2] + t * (1 - blue[2]))
    else:
        return (1 - v * (1 - red[0]), 1 - v * (1 - red[1]),
                1 - v * (1 - red[2]))


# Molecule attribution drawing


def draw_mol_attribution(
    mol,
    atom_attrs: np.ndarray,
    title: str,
    score: float,
    save_path: str,
    task_name: str = "PCE",
    show_top_n: int = 5,
):
    """Draw a molecule with atom attribution heatmap, colorbar, and top-N annotations.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    atom_attrs : np.ndarray
        Raw attribution values per atom.
    title : str
        Figure title prefix.
    score : float
        Predicted score to display.
    save_path : str
        Output PNG path.
    task_name : str
        Task name for title (default from TASK_NAMES[0]).
    show_top_n : int
        Number of top atoms to annotate.
    """
    max_abs = np.abs(atom_attrs).max()
    normed = atom_attrs / (max_abs + 1e-8)

    atom_colors = {i: attr_to_rgb(normed[i]) for i in range(len(normed))}
    highlight_atoms = list(range(len(normed)))

    highlight_bonds = []
    bond_colors = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 < len(normed) and a2 < len(normed):
            avg_v = (normed[a1] + normed[a2]) / 2.0
            highlight_bonds.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = attr_to_rgb(avg_v)

    drawer = rdMolDraw2D.MolDraw2DSVG(600, 450)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [4, 1]},
    )

    mol_img = None
    try:
        png_data = _svg2png(bytestring=svg_text.encode())
        mol_img = Image.open(io.BytesIO(png_data))
    except Exception:
        mol_img = None

    if mol_img is None:
        mol_img = Draw.MolToImage(
            mol,
            size=(600, 450),
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
        )

    axes[0].imshow(mol_img)
    axes[0].axis("off")
    axes[0].set_title(
        f"{title}\nPredicted {task_name} score: {score:.4f}",
        fontsize=12,
        fontweight="bold",
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "attr", ["#1f77b4", "white", "#d62728"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], orientation="vertical", fraction=0.8)
    cbar.set_label("Normalized Attribution", fontsize=10)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(
        ["−1\n(negative)", "−0.5", "0", "+0.5", "+1\n(positive)"])
    axes[1].axis("off")

    top_n = np.argsort(np.abs(atom_attrs))[-show_top_n:]
    text_lines = []
    for idx in top_n:
        text_lines.append(f"Atom {idx}: {atom_attrs[idx]:+.3f}")
    if text_lines:
        axes[0].text(
            0.02,
            0.02,
            "\n".join(text_lines),
            color="black",
            fontsize=8,
            transform=axes[0].transAxes,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  PNG → {save_path}")


# Feature attribution bar chart


def draw_feature_attribution(
    attrs: np.ndarray,
    names: list,
    save_path: str,
    title: str,
    score: float,
    task_name: str = "PCE",
):
    """Draw a horizontal bar chart of extra feature attributions.

    Parameters
    ----------
    attrs : np.ndarray
        Attribution values per feature.
    names : list
        Feature names.
    save_path : str
        Output PNG path.
    title : str
        Figure title prefix.
    score : float
        Predicted score to display.
    task_name : str
        Task name for title.
    """
    order = np.argsort(attrs)
    sorted_attrs = attrs[order]
    sorted_names = [names[i] for i in order]
    colors = ["#1f77b4" if v < 0 else "#d62728" for v in sorted_attrs]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45)))
    bars = ax.barh(
        range(len(sorted_attrs)),
        sorted_attrs,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Integrated Gradient Attribution", fontsize=11)
    ax.set_title(
        f"{title} — Extra Feature Attribution\n"
        f"Task: {task_name} | Predicted Score: {score:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = np.abs(sorted_attrs).max() if len(sorted_attrs) > 0 else 1.0
    ax.set_xlim(-x_max * 1.25, x_max * 1.25)
    offset = x_max * 0.005
    for bar, val in zip(bars, sorted_attrs):
        if abs(val) < 1e-8:
            continue
        ax.text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            color="black",
        )

    pos_patch = mpatches.Patch(color="#d62728", label="Positive contribution")
    neg_patch = mpatches.Patch(color="#1f77b4", label="Negative contribution")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Feature bar chart → {save_path}")


# Score ranking bar chart (unified)


def draw_score_ranking(
    name_score_pairs: list,
    save_dir: str,
    title_suffix: str = "",
    filename: str = "score_ranking.png",
):
    """Draw a horizontal bar chart ranking materials by predicted score.

    Unified from ``_draw_score_ranking`` (htl_ranking.py) and
    ``_draw_chained_score_ranking`` (htl_chain_attr.py).

    Parameters
    ----------
    name_score_pairs : list of (str, float)
        (material_name, score) pairs.
    save_dir : str
        Output directory.
    title_suffix : str
        Optional text appended to the title (e.g. "  [Chained Attribution]").
    filename : str
        Output PNG filename.
    """
    pairs = sorted(name_score_pairs, key=lambda x: x[1])
    names = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(scores)))

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(f"Predicted {TASK_NAMES[0]} Score", fontsize=11)
    ax.set_title(
        f"Material Score Ranking{title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_range = max(scores) - min(scores) if max(scores) != min(scores) else 1.0
    for bar, v in zip(bars, scores):
        ax.text(v + x_range * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}",
                va="center",
                fontsize=9)

    plt.tight_layout()
    path = str(Path(save_dir) / filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Score ranking chart → {path}")


# CSV merging (unified)


def merge_csvs(
    save_dir: str,
    pattern: str,
    output_name: str,
):
    """Merge all CSV files matching *pattern* in *save_dir* into a single CSV.

    Unified from ``_merge_summary_csvs`` and ``_merge_diff_csvs``.

    Parameters
    ----------
    save_dir : str
        Directory containing the CSVs.
    pattern : str
        Glob pattern, e.g. ``*_summary.csv`` or ``*_diff_summary.csv``.
    output_name : str
        Merged output filename, e.g. ``all_attributions.csv``.
    """
    csv_files = [str(f) for f in Path(save_dir).glob(pattern)]
    if not csv_files:
        return
    merged = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    out = str(Path(save_dir) / output_name)
    merged.to_csv(out, index=False)
    logger.info(f"Merged → {out}")


# Molecule → PIL Image (for diff attribution)


def mol_to_image(mol, atom_attrs: np.ndarray) -> Image.Image:
    """Render one molecule as a PIL Image with attribution heatmap.

    Extracted from DiffAttrExplainer._mol_to_image.
    """
    max_abs = np.abs(atom_attrs).max() + 1e-8
    normed = atom_attrs / max_abs
    n_atoms = len(normed)

    atom_colors = {i: attr_to_rgb(normed[i]) for i in range(n_atoms)}
    h_atoms = list(range(n_atoms))
    h_bonds, b_colors = [], {}

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 < n_atoms and a2 < n_atoms:
            avg = (normed[a1] + normed[a2]) / 2.0
            h_bonds.append(bond.GetIdx())
            b_colors[bond.GetIdx()] = attr_to_rgb(avg)

    drawer = rdMolDraw2D.MolDraw2DSVG(520, 400)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=h_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=h_bonds,
        highlightBondColors=b_colors,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    try:
        png_data = _svg2png(bytestring=svg_text.encode())
        return Image.open(io.BytesIO(png_data))
    except Exception:
        return Draw.MolToImage(mol,
                               size=(520, 400),
                               highlightAtoms=h_atoms,
                               highlightAtomColors=atom_colors)
