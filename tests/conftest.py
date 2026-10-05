import sys
from unittest.mock import MagicMock

_mock_chemprop = MagicMock()
_mock_chemprop_data = MagicMock()
_mock_chemprop_data_collate = MagicMock()
_mock_chemprop_data_collate.BatchMolGraph = MagicMock
_mock_chemprop_data_collate.collate_batch = MagicMock(return_value=None)

_mock_chemprop_nn = MagicMock()
_mock_chemprop_nn.BondMessagePassing = MagicMock
_mock_chemprop_nn.MeanAggregation = MagicMock
_mock_chemprop_nn.SumAggregation = MagicMock
_mock_chemprop_nn.NormAggregation = MagicMock

_mock_chemprop_featurizers = MagicMock()
_mock_chemprop_featurizers.SimpleMoleculeMolGraphFeaturizer = MagicMock
_mock_chemprop_featurizers_atom = MagicMock()
_mock_chemprop_featurizers_atom.MultiHotAtomFeaturizer = MagicMock()
_mock_chemprop_featurizers_atom.MultiHotAtomFeaturizer.v2 = MagicMock(return_value=MagicMock())
_mock_chemprop_featurizers_bond = MagicMock()
_mock_chemprop_featurizers_bond.MultiHotBondFeaturizer = MagicMock(return_value=MagicMock())

_mock_chemprop.data = _mock_chemprop_data
_mock_chemprop.data.collate = _mock_chemprop_data_collate
_mock_chemprop.nn = _mock_chemprop_nn
_mock_chemprop.featurizers = _mock_chemprop_featurizers
_mock_chemprop.featurizers.atom = _mock_chemprop_featurizers_atom
_mock_chemprop.featurizers.bond = _mock_chemprop_featurizers_bond

sys.modules["chemprop"] = _mock_chemprop
sys.modules["chemprop.data"] = _mock_chemprop_data
sys.modules["chemprop.data.collate"] = _mock_chemprop_data_collate
sys.modules["chemprop.nn"] = _mock_chemprop_nn
sys.modules["chemprop.featurizers"] = _mock_chemprop_featurizers
sys.modules["chemprop.featurizers.atom"] = _mock_chemprop_featurizers_atom
sys.modules["chemprop.featurizers.bond"] = _mock_chemprop_featurizers_bond
