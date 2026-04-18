"""htl_package — shared modules for HTL prediction pipeline.

Convenience import hub: import all public symbols from submodules.
"""

# constants
from .constants import (
    EXTRA_COLS,
    EXTRA_DIM,
    GLOBAL_COLS,
    GLOBAL_DIM,
    TASK_NAMES,
    NUM_TASKS,
    logger,
    DEVICE,
)

# configs
from .configs import (
    ModelConfig,
    TrainingConfig,
    FinetuneConfig,
    PredictConfig,
    SurrogateConfig,
)

# features
from .features import (
    _extra_feat,
    _extra_feat_single,
    _global_feat,
    load_and_preprocess,
)

# datasets
from .datasets import (
    PairDataset,
    ListDataset,
    CachedPairDataset,
    SurrogateDataset,
    collate_fn,
    collate_list_fn,
    collate_cached_batch,
    _collate_surrogate,
)

# models
from .models import (
    DMPNNEncoder,
    HTLRankingModel,
    MultiTaskRankingLoss,
    EarlyStopping,
    SurrogateModel,
)

# checkpoint
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model_for_inference,
)

# visualization
from .visualization import (
    attr_to_rgb,
    draw_mol_attribution,
    draw_feature_attribution,
    draw_score_ranking,
    merge_csvs,
    mol_to_image,
)

# explainer
from .explainer import (
    IGExplainer,
    DiffAttrExplainer,
    explain,
    diff_attr,
)

__all__ = [
    # constants
    "EXTRA_COLS", "EXTRA_DIM", "GLOBAL_COLS", "GLOBAL_DIM",
    "TASK_NAMES", "NUM_TASKS", "logger", "DEVICE",
    # configs
    "ModelConfig", "TrainingConfig", "FinetuneConfig", "PredictConfig",
    "SurrogateConfig",
    # features
    "_extra_feat", "_extra_feat_single", "_global_feat", "load_and_preprocess",
    # datasets
    "PairDataset", "ListDataset", "CachedPairDataset", "SurrogateDataset",
    "collate_fn", "collate_list_fn", "collate_cached_batch", "_collate_surrogate",
    # models
    "DMPNNEncoder", "HTLRankingModel", "MultiTaskRankingLoss",
    "EarlyStopping", "SurrogateModel",
    # checkpoint
    "save_checkpoint", "load_checkpoint", "load_model_for_inference",
    # visualization
    "attr_to_rgb", "draw_mol_attribution", "draw_feature_attribution",
    "draw_score_ranking", "merge_csvs", "mol_to_image",
    # explainer
    "IGExplainer", "DiffAttrExplainer", "explain", "diff_attr",
]
