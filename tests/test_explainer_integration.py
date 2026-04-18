"""
Integration tests for explainer consistency between Explain and Diff Attribution modes.

Run directly (not via pytest with conftest.py mock):
    conda activate chemprop2
    python tests/test_explainer_integration.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from htl_package.explainer import IGExplainer, DiffAttrExplainer
from htl_package.models import HTLRankingModel
from htl_package.constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS, DEVICE

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from rdkit import Chem

import tempfile


def _make_model():
    return HTLRankingModel(
        hidden_size=64,
        depth=2,
        dropout=0.1,
        ffn_hidden=32,
        extra_dim=EXTRA_DIM,
        global_dim=GLOBAL_DIM,
        num_tasks=NUM_TASKS,
    ).to(DEVICE)


def _make_scaler():
    scaler = StandardScaler()
    scaler.fit(np.random.randn(20, EXTRA_DIM).astype(np.float32))
    return scaler


def test_joint_and_separate_produce_different_extra_ig():
    """Joint IG and separate IG should produce DIFFERENT extra feature attributions.

    This test confirms the bug existed: if they were the same, there would be
    no inconsistency between explain and diff_attr modes.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    model = _make_model()
    scaler = _make_scaler()
    ig = IGExplainer(model=model, scaler=scaler, n_steps=10)

    mol = Chem.MolFromSmiles("CCO")
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=MultiHotAtomFeaturizer.v2(),
        bond_featurizer=MultiHotBondFeaturizer(),
    )

    ef_raw = np.random.randn(EXTRA_DIM).astype(np.float32)
    ef = torch.tensor(
        scaler.transform(ef_raw.reshape(1, -1)), dtype=torch.float32
    ).to(DEVICE)
    gf = torch.zeros(1, GLOBAL_DIM, dtype=torch.float32).to(DEVICE)

    bmg_joint = BatchMolGraph([featurizer(mol)])
    _, extra_joint, _ = ig._joint_ig(bmg_joint, ef, gf)

    bmg_atom = BatchMolGraph([featurizer(mol)])
    ig._atom_ig(bmg_atom, ef, gf)

    bmg_extra = BatchMolGraph([featurizer(mol)])
    extra_separate = ig._extra_ig(bmg_extra, ef, gf)

    assert not np.allclose(extra_joint, extra_separate, atol=1e-3), (
        "Joint IG and separate IG should produce different extra feature "
        "attributions for a non-trivial model."
    )
    print("  PASS: joint IG != separate IG (confirms the bug existed)")


def test_explain_and_diff_attr_consistent_for_same_molecule():
    """After the fix, explain mode and diff_attr mode should produce the
    SAME extra feature and atom attributions for the same molecule."""
    torch.manual_seed(42)
    np.random.seed(42)
    model = _make_model()
    scaler = _make_scaler()
    n_steps = 10

    ig_explainer = IGExplainer(model=model, scaler=scaler, n_steps=n_steps)
    diff_explainer = DiffAttrExplainer(
        model=model, scaler=scaler, n_steps=n_steps
    )

    smiles = "CCO"
    ef_raw = np.random.randn(EXTRA_DIM).astype(np.float32)
    ef_scaled = scaler.transform(ef_raw.reshape(1, -1))[0]
    gf = np.zeros(GLOBAL_DIM, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        explain_result = ig_explainer.explain_molecule(
            smiles=smiles,
            ef_scaled=ef_scaled,
            gf=gf,
            material_name="test_mol",
            save_dir=tmpdir,
            use_joint_ig=True,
        )

    diff_result = diff_explainer._score_and_attrs(
        smiles=smiles,
        ef_scaled=ef_scaled,
        gf=gf,
    )

    np.testing.assert_allclose(
        explain_result["extra_attrs"],
        diff_result["extra_attrs"],
        atol=1e-5,
        err_msg="Explain mode (joint IG) and diff_attr mode should "
        "produce the same extra feature attributions for the same molecule",
    )
    print("  PASS: extra_attrs consistent between explain and diff_attr")

    np.testing.assert_allclose(
        explain_result["atom_attrs"],
        diff_result["atom_attrs"],
        atol=1e-5,
        err_msg="Explain mode (joint IG) and diff_attr mode should "
        "produce the same atom attributions for the same molecule",
    )
    print("  PASS: atom_attrs consistent between explain and diff_attr")


def test_explain_and_diff_attr_same_score():
    """Both modes should produce the same predicted score for the same molecule."""
    torch.manual_seed(42)
    np.random.seed(42)
    model = _make_model()
    scaler = _make_scaler()

    ig_explainer = IGExplainer(model=model, scaler=scaler, n_steps=10)
    diff_explainer = DiffAttrExplainer(
        model=model, scaler=scaler, n_steps=10
    )

    smiles = "CCO"
    ef_raw = np.random.randn(EXTRA_DIM).astype(np.float32)
    ef_scaled = scaler.transform(ef_raw.reshape(1, -1))[0]
    gf = np.zeros(GLOBAL_DIM, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        explain_result = ig_explainer.explain_molecule(
            smiles=smiles,
            ef_scaled=ef_scaled,
            gf=gf,
            material_name="test_mol",
            save_dir=tmpdir,
        )

    diff_result = diff_explainer._score_and_attrs(
        smiles=smiles,
        ef_scaled=ef_scaled,
        gf=gf,
    )

    assert abs(explain_result["score"] - diff_result["score"]) < 1e-5
    print("  PASS: scores consistent between explain and diff_attr")


def test_completeness_axiom():
    """Joint IG should satisfy the completeness axiom:
    sum of all IG attributions ≈ score(input) - score(baseline=0)"""
    torch.manual_seed(42)
    np.random.seed(42)
    model = _make_model()
    scaler = _make_scaler()
    ig = IGExplainer(model=model, scaler=scaler, n_steps=200)

    mol = Chem.MolFromSmiles("CCO")
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=MultiHotAtomFeaturizer.v2(),
        bond_featurizer=MultiHotBondFeaturizer(),
    )

    ef_raw = np.random.randn(EXTRA_DIM).astype(np.float32)
    ef = torch.tensor(
        scaler.transform(ef_raw.reshape(1, -1)), dtype=torch.float32
    ).to(DEVICE)
    gf = torch.zeros(1, GLOBAL_DIM, dtype=torch.float32).to(DEVICE)

    bmg = BatchMolGraph([featurizer(mol)])

    with torch.no_grad():
        score_input = model.encode(bmg, ef, gf)[0, 0].item()

    bmg_ig = BatchMolGraph([featurizer(mol)])
    atom_ig, extra_ig, global_ig = ig._joint_ig(bmg_ig, ef, gf)

    total_ig = float(atom_ig.sum()) + float(extra_ig.sum()) + float(
        global_ig.sum()
    )

    np.testing.assert_allclose(
        total_ig,
        score_input,
        atol=0.1,
        err_msg="Joint IG should satisfy the completeness axiom: "
        "sum of all IG attributions ≈ score(input) - score(baseline=0)",
    )
    print("  PASS: completeness axiom satisfied")


if __name__ == "__main__":
    tests = [
        test_joint_and_separate_produce_different_extra_ig,
        test_explain_and_diff_attr_consistent_for_same_molecule,
        test_explain_and_diff_attr_same_score,
        test_completeness_axiom,
    ]

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        print(f"\nRunning: {name}")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
