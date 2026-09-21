import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, call
from sklearn.preprocessing import StandardScaler

from htl_package.explainer import IGExplainer, DiffAttrExplainer
from htl_package.models import HTLRankingModel
from htl_package.constants import EXTRA_DIM, GLOBAL_DIM, NUM_TASKS, DEVICE


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


def _mock_score_and_attrs_env(explainer, n_atoms=3):
    joint_return = (
        np.random.randn(n_atoms + 2),
        np.random.randn(EXTRA_DIM),
        np.random.randn(GLOBAL_DIM),
    )
    explainer._ig._joint_ig = MagicMock(return_value=joint_return)
    explainer._ig._atom_ig = MagicMock(
        return_value=np.random.randn(n_atoms + 2)
    )
    explainer._ig._extra_ig = MagicMock(
        return_value=np.random.randn(EXTRA_DIM)
    )

    mock_mol = MagicMock()
    mock_mol.GetNumAtoms.return_value = n_atoms

    return mock_mol, joint_return


class TestScoreAndAttrsUsesJointIG:
    def test_joint_ig_called_not_separate(self):
        model = _make_model()
        scaler = _make_scaler()
        explainer = DiffAttrExplainer(model=model, scaler=scaler, n_steps=5)

        mock_mol, _ = _mock_score_and_attrs_env(explainer, n_atoms=3)

        with patch("htl_package.explainer.Chem.MolFromSmiles", return_value=mock_mol), \
             patch("htl_package.explainer.AllChem.Compute2DCoords"), \
             patch("htl_package.explainer.BatchMolGraph"), \
             patch.object(model, "encode", return_value=torch.tensor([[1.0]])):
            explainer._score_and_attrs(
                "CCO",
                np.random.randn(EXTRA_DIM).astype(np.float32),
                np.random.randn(GLOBAL_DIM).astype(np.float32),
            )

        explainer._ig._joint_ig.assert_called_once()
        explainer._ig._atom_ig.assert_not_called()
        explainer._ig._extra_ig.assert_not_called()

    def test_returns_joint_ig_extra_attrs(self):
        model = _make_model()
        scaler = _make_scaler()
        explainer = DiffAttrExplainer(model=model, scaler=scaler, n_steps=5)

        mock_mol, joint_return = _mock_score_and_attrs_env(explainer, n_atoms=3)
        expected_extra = joint_return[1]

        with patch("htl_package.explainer.Chem.MolFromSmiles", return_value=mock_mol), \
             patch("htl_package.explainer.AllChem.Compute2DCoords"), \
             patch("htl_package.explainer.BatchMolGraph"), \
             patch.object(model, "encode", return_value=torch.tensor([[1.0]])):
            result = explainer._score_and_attrs(
                "CCO",
                np.random.randn(EXTRA_DIM).astype(np.float32),
                np.random.randn(GLOBAL_DIM).astype(np.float32),
            )

        np.testing.assert_array_equal(result["extra_attrs"], expected_extra)

    def test_atom_attrs_truncated_to_num_atoms(self):
        model = _make_model()
        scaler = _make_scaler()
        explainer = DiffAttrExplainer(model=model, scaler=scaler, n_steps=5)

        n_atoms = 3
        mock_mol, joint_return = _mock_score_and_attrs_env(
            explainer, n_atoms=n_atoms
        )
        expected_atom = joint_return[0][:n_atoms]

        with patch("htl_package.explainer.Chem.MolFromSmiles", return_value=mock_mol), \
             patch("htl_package.explainer.AllChem.Compute2DCoords"), \
             patch("htl_package.explainer.BatchMolGraph"), \
             patch.object(model, "encode", return_value=torch.tensor([[1.0]])):
            result = explainer._score_and_attrs(
                "CCO",
                np.random.randn(EXTRA_DIM).astype(np.float32),
                np.random.randn(GLOBAL_DIM).astype(np.float32),
            )

        assert len(result["atom_attrs"]) == n_atoms
        np.testing.assert_array_equal(result["atom_attrs"], expected_atom)

    def test_score_from_model_encode(self):
        model = _make_model()
        scaler = _make_scaler()
        explainer = DiffAttrExplainer(model=model, scaler=scaler, n_steps=5)

        mock_mol, _ = _mock_score_and_attrs_env(explainer, n_atoms=3)

        expected_score = 2.718
        with patch("htl_package.explainer.Chem.MolFromSmiles", return_value=mock_mol), \
             patch("htl_package.explainer.AllChem.Compute2DCoords"), \
             patch("htl_package.explainer.BatchMolGraph"), \
             patch.object(model, "encode", return_value=torch.tensor([[expected_score]])):
            result = explainer._score_and_attrs(
                "CCO",
                np.random.randn(EXTRA_DIM).astype(np.float32),
                np.random.randn(GLOBAL_DIM).astype(np.float32),
            )

        assert abs(result["score"] - expected_score) < 1e-5


class TestExplainMoleculeDefaultUsesJointIG:
    def test_explain_molecule_default_calls_joint_ig(self):
        model = _make_model()
        scaler = _make_scaler()
        explainer = IGExplainer(model=model, scaler=scaler, n_steps=5)

        joint_return = (
            np.random.randn(5),
            np.random.randn(EXTRA_DIM),
            np.random.randn(GLOBAL_DIM),
        )
        explainer._joint_ig = MagicMock(return_value=joint_return)
        explainer._atom_ig = MagicMock(return_value=np.random.randn(5))
        explainer._extra_ig = MagicMock(return_value=np.random.randn(EXTRA_DIM))

        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 3

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("htl_package.explainer.Chem.MolFromSmiles", return_value=mock_mol), \
             patch("htl_package.explainer.AllChem.Compute2DCoords"), \
             patch("htl_package.explainer.BatchMolGraph"), \
             patch.object(model, "encode", return_value=torch.tensor([[1.0]])), \
             patch("htl_package.explainer.draw_mol_attribution"), \
             patch("htl_package.explainer.draw_feature_attribution"):
            explainer.explain_molecule(
                smiles="CCO",
                ef_scaled=np.random.randn(EXTRA_DIM).astype(np.float32),
                gf=np.zeros(GLOBAL_DIM, dtype=np.float32),
                material_name="test",
                save_dir=tmpdir,
            )

        explainer._joint_ig.assert_called_once()
        explainer._atom_ig.assert_not_called()
        explainer._extra_ig.assert_not_called()

    def test_explain_molecule_use_joint_ig_false_uses_separate(self):
        model = _make_model()
        scaler = _make_scaler()
        explainer = IGExplainer(model=model, scaler=scaler, n_steps=5)

        explainer._joint_ig = MagicMock(return_value=(
            np.random.randn(5),
            np.random.randn(EXTRA_DIM),
            np.random.randn(GLOBAL_DIM),
        ))
        explainer._atom_ig = MagicMock(return_value=np.random.randn(5))
        explainer._extra_ig = MagicMock(return_value=np.random.randn(EXTRA_DIM))

        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 3

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("htl_package.explainer.Chem.MolFromSmiles", return_value=mock_mol), \
             patch("htl_package.explainer.AllChem.Compute2DCoords"), \
             patch("htl_package.explainer.BatchMolGraph"), \
             patch.object(model, "encode", return_value=torch.tensor([[1.0]])), \
             patch("htl_package.explainer.draw_mol_attribution"), \
             patch("htl_package.explainer.draw_feature_attribution"):
            explainer.explain_molecule(
                smiles="CCO",
                ef_scaled=np.random.randn(EXTRA_DIM).astype(np.float32),
                gf=np.zeros(GLOBAL_DIM, dtype=np.float32),
                material_name="test",
                save_dir=tmpdir,
                use_joint_ig=False,
            )

        explainer._joint_ig.assert_not_called()
        explainer._atom_ig.assert_called_once()
        explainer._extra_ig.assert_called_once()


class TestConsistencyBetweenModes:
    def test_both_modes_use_same_ig_method(self):
        model = _make_model()
        scaler = _make_scaler()

        ig_explainer = IGExplainer(model=model, scaler=scaler, n_steps=5)
        diff_explainer = DiffAttrExplainer(
            model=model, scaler=scaler, n_steps=5
        )

        ig_explainer._joint_ig = MagicMock(return_value=(
            np.random.randn(5),
            np.random.randn(EXTRA_DIM),
            np.random.randn(GLOBAL_DIM),
        ))
        diff_explainer._ig._joint_ig = MagicMock(return_value=(
            np.random.randn(5),
            np.random.randn(EXTRA_DIM),
            np.random.randn(GLOBAL_DIM),
        ))

        assert hasattr(diff_explainer._ig, "_joint_ig")
        assert diff_explainer._ig._joint_ig is not None

    def test_diff_attr_explainer_uses_ig_explainer_internally(self):
        model = _make_model()
        scaler = _make_scaler()

        diff_explainer = DiffAttrExplainer(
            model=model, scaler=scaler, n_steps=5
        )

        assert isinstance(diff_explainer._ig, IGExplainer)
        assert diff_explainer._ig.n_steps == 5
        assert diff_explainer._ig.target_task == 0


try:
    from chemprop.data.collate import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.featurizers.atom import MultiHotAtomFeaturizer
    from chemprop.featurizers.bond import MultiHotBondFeaturizer
    from rdkit import Chem

    _HAS_REAL_CHEMPROP = True
except Exception:
    _HAS_REAL_CHEMPROP = False


@pytest.mark.skipif(not _HAS_REAL_CHEMPROP, reason="chemprop not available")
class TestIntegrationJointIGVsSeparateIG:
    @pytest.fixture
    def model_and_scaler(self):
        torch.manual_seed(42)
        np.random.seed(42)
        model = _make_model()
        scaler = _make_scaler()
        return model, scaler

    def test_joint_and_separate_produce_different_extra_ig(
        self, model_and_scaler
    ):
        model, scaler = model_and_scaler
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

    def test_explain_and_diff_attr_consistent_for_same_molecule(
        self, model_and_scaler, tmp_path
    ):
        model, scaler = model_and_scaler
        n_steps = 10

        ig_explainer = IGExplainer(
            model=model, scaler=scaler, n_steps=n_steps
        )
        diff_explainer = DiffAttrExplainer(
            model=model, scaler=scaler, n_steps=n_steps
        )

        smiles = "CCO"
        ef_raw = np.random.randn(EXTRA_DIM).astype(np.float32)
        ef_scaled = scaler.transform(ef_raw.reshape(1, -1))[0]
        gf = np.zeros(GLOBAL_DIM, dtype=np.float32)

        explain_result = ig_explainer.explain_molecule(
            smiles=smiles,
            ef_scaled=ef_scaled,
            gf=gf,
            material_name="test_mol",
            save_dir=str(tmp_path),
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

        np.testing.assert_allclose(
            explain_result["atom_attrs"],
            diff_result["atom_attrs"],
            atol=1e-5,
            err_msg="Explain mode (joint IG) and diff_attr mode should "
            "produce the same atom attributions for the same molecule",
        )

    def test_explain_and_diff_attr_same_score(self, model_and_scaler, tmp_path):
        model, scaler = model_and_scaler

        ig_explainer = IGExplainer(model=model, scaler=scaler, n_steps=10)
        diff_explainer = DiffAttrExplainer(
            model=model, scaler=scaler, n_steps=10
        )

        smiles = "CCO"
        ef_raw = np.random.randn(EXTRA_DIM).astype(np.float32)
        ef_scaled = scaler.transform(ef_raw.reshape(1, -1))[0]
        gf = np.zeros(GLOBAL_DIM, dtype=np.float32)

        explain_result = ig_explainer.explain_molecule(
            smiles=smiles,
            ef_scaled=ef_scaled,
            gf=gf,
            material_name="test_mol",
            save_dir=str(tmp_path),
        )

        diff_result = diff_explainer._score_and_attrs(
            smiles=smiles,
            ef_scaled=ef_scaled,
            gf=gf,
        )

        assert abs(explain_result["score"] - diff_result["score"]) < 1e-5


@pytest.mark.skipif(not _HAS_REAL_CHEMPROP, reason="chemprop not available")
class TestIntegrationJointIGCompleteness:
    def test_completeness_axiom(self):
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
