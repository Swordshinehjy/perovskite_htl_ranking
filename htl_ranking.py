"""
Perovskite HTL Prediction
=====================================
Pipeline:
  1. chemprop v2 D-MPNN encoding + extra feature concatenation
  2. Pairwise Ranking Loss (margin ranking + delta regression)

Modes:
  train / finetune / predict / list_rank / explain / diff_attr
"""

from htl_package.configs import ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig
from htl_package.training import train, finetune
from htl_package.prediction import predict_batch, predict_list
from htl_package.explainer import explain, diff_attr

if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser(description="HTL Pairwise Ranking via D-MPNN")
    p.add_argument(
        "--mode",
        choices=[
            "train", "predict", "finetune", "list_rank", "explain", 
            "diff_attr"
        ],
        default="train",
    )
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--predict_csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--checkpoint_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--ffn_hidden", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument(
        "--early_stop_warmup",
        type=int,
        default=None,
        help="Number of initial epochs to skip before early stopping kicks in. "
             "Prevents a spurious low val loss on small val sets (common with "
             "group/LOGO splits) from locking in a bad checkpoint. Default: 20.",
    )
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["random", "group"],
        help="Split strategy: 'random' (default) or 'group' (doi-based group split)",
    )
    p.add_argument(
        "--n_cv_folds",
        type=int,
        default=None,
        help="Number of LOGO CV folds (only with --split group). "
             "Uses the N largest doi groups as held-out test sets.",
    )
    p.add_argument("--finetune_epochs", type=int, default=None)
    p.add_argument("--finetune_lr", type=float, default=None)
    p.add_argument("--explain_csv", type=str, default=None)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--explain_dir", type=str, default="explain_output")
    p.add_argument("--analyze_csv", type=str, default=None)
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument(
        "--diff_csv",
        type=str,
        default=None,
        help="Input CSV for diff_attr mode (pair format, same as htl-new.csv)",
    )
    p.add_argument(
        "--diff_dir",
        type=str,
        default="diff_output",
        help="Output directory for diff_attr mode",
    )
    args = p.parse_args()

    def merge_args(config_cls, args, arg_mapping: dict):
        kwargs = {}
        for config_field, arg_name in arg_mapping.items():
            arg_val = getattr(args, arg_name)
            if arg_val is not None:
                kwargs[config_field] = arg_val
        return config_cls(**kwargs)

    if args.mode == "train":
        model_config = merge_args(
            ModelConfig, args, {
                "hidden_size": "hidden_size",
                "depth": "depth",
                "dropout": "dropout",
                "ffn_hidden": "ffn_hidden",
            })
        train_config = merge_args(
            TrainingConfig, args, {
                "csv_path": "csv",
                "save_dir": "save_dir",
                "epochs": "epochs",
                "batch_size": "batch_size",
                "lr": "lr",
                "weight_decay": "weight_decay",
                "patience": "patience",
                "early_stop_warmup": "early_stop_warmup",
                "val_ratio": "val_ratio",
                "test_ratio": "test_ratio",
                "margin": "margin",
                "seed": "seed",
                "split": "split",
                "n_cv_folds": "n_cv_folds",
            })
        results = train(model_config, train_config)
        if "cv_fold_metrics" in results and results["cv_fold_metrics"]:
            print("\n===== LOGO CV Aggregated Metrics =====")
            for k, v in results["test_metrics"].items():
                print(f"  {k:35s}: {v:.4f}")
            print(f"\n  ({len(results['cv_fold_metrics'])} folds)")
        else:
            print("\n===== Final Test Metrics =====")
            for k, v in results["test_metrics"].items():
                print(f"  {k:30s}: {v:.4f}")

    elif args.mode == "finetune":
        finetune_config = merge_args(
            FinetuneConfig, args, {
                "csv_path": "csv",
                "checkpoint_dir": "checkpoint_dir",
                "checkpoint_name": "checkpoint_name",
                "save_dir": "save_dir",
                "finetune_epochs": "finetune_epochs",
                "batch_size": "batch_size",
                "lr": "finetune_lr",
                "margin": "margin",
                "seed": "seed",
            })
        results = finetune(finetune_config)
        print(
            f"\n===== Final model saved to {results['final_checkpoint_dir']}/{results['final_checkpoint_name']} ====="
        )

    elif args.mode == "predict":
        if not args.predict_csv:
            p.error("--predict_csv is required for predict mode")
        predict_config = merge_args(
            PredictConfig, args, {
                "predict_csv": "predict_csv",
                "checkpoint_dir": "checkpoint_dir",
                "checkpoint_name": "checkpoint_name",
                "output_path": "output",
            })
        df_new = pd.read_csv(predict_config.predict_csv)
        result = predict_batch(df_new=df_new,
                               checkpoint_dir=predict_config.checkpoint_dir,
                               checkpoint_name=predict_config.checkpoint_name,
                               output_path=predict_config.output_path)
        print(result[["Materials_1", "Materials_2", "preferred_PCE"]])

    elif args.mode == "list_rank":
        if not args.predict_csv:
            p.error("--predict_csv is required for list_rank mode")
        df_list = pd.read_csv(args.predict_csv)
        result = predict_list(
            df_list=df_list,
            checkpoint_dir=args.checkpoint_dir or "checkpoints",
            checkpoint_name=args.checkpoint_name or "final_model",
            output_path=args.output,
            batch_size=args.batch_size or 32,
        )
        print(result[["rank", "Materials", "score_PCE"]])

    elif args.mode == "explain":
        if not args.explain_csv:
            p.error("--explain_csv is required for explain mode")
        if not args.checkpoint_dir:
            p.error("--checkpoint_dir is required for explain mode")
        df_explain = pd.read_csv(args.explain_csv)
        results = explain(
            df_list=df_explain,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name or "final_model",
            save_dir=args.explain_dir or "explain_output",
            n_steps=args.n_steps or 50,
        )
        print(f"\n===== Explain Done: {len(results)} materials =====")
        for r in results:
            top_feat = r["extra_names"][int(np.argmax(r["extra_attrs"]))]
            print(f"  {r['material_name']:30s} "
                  f"score={r['score']:.4f}  "
                  f"top_feature={top_feat}")

    elif args.mode == "diff_attr":
        if not args.diff_csv:
            p.error("--diff_csv is required for diff_attr mode")
        if not args.checkpoint_dir:
            p.error("--checkpoint_dir is required for diff_attr mode")

        df_diff = pd.read_csv(args.diff_csv)
        results = diff_attr(
            df_pairs=df_diff,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name or "final_model",
            save_dir=args.diff_dir or "diff_output",
            n_steps=args.n_steps or 50,
        )

        print(f"\n===== Diff Attribution Done: {len(results)} pairs =====")
        print(
            f"{'Pair':<50} {'Δscore':>9}  {'Preferred':<20}  {'Top Δfeature (val)'}"
        )
        print("-" * 105)
        for r in results:
            best_feat_idx = int(np.argmax(np.abs(r["delta_extra"])))
            best_feat = r["extra_names"][best_feat_idx]
            best_val = r["delta_extra"][best_feat_idx]
            print(f"  {r['pair_id'][:48]:<50} "
                  f"{r['score_diff']:+9.4f}  "
                  f"{r['preferred']:<20}  "
                  f"{best_feat} ({best_val:+.4f})")

"""
# Examples

# Training (random split, default)
python htl_ranking.py --mode train --csv htl-data-combinations.csv

# Training (group split by doi, single split)
python htl_ranking.py --mode train --csv htl-data-combinations.csv --split group

# Training (group split + Leave-One-Group-Out CV over top-5 largest doi groups)
python htl_ranking.py --mode train --csv htl-data-combinations.csv --split group --n_cv_folds 5

# Fine-tuning
python htl_ranking.py --mode finetune --csv htl-data-combinations.csv --checkpoint_dir checkpoints --checkpoint_name best_model --finetune_epochs 10 --finetune_lr 1e-5

# Pairwise prediction
python htl_ranking.py --mode predict --predict_csv htl-new.csv --checkpoint_dir checkpoints --checkpoint_name final_model

# List ranking
python htl_ranking.py --mode list_rank --predict_csv ranking-new.csv --checkpoint_dir checkpoints --checkpoint_name final_model --output ranked_results.csv

# Explain (single-molecule IG attribution)
python htl_ranking.py --mode explain --explain_csv ranking-new.csv --checkpoint_dir checkpoints --checkpoint_name final_model --explain_dir explain_output --n_steps 100

# Differential attribution
python htl_ranking.py --mode diff_attr --diff_csv htl-new.csv --checkpoint_dir checkpoints --checkpoint_name final_model --diff_dir diff_output --n_steps 100
"""
