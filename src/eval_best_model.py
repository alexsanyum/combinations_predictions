# evaluate_best_models.py
import argparse
import os
import gc
from glob import glob
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from src.imcp import imcp_score 
from src.cluster_undersampling import ClusterUnderSampler

def imcp_score_adapted(y_true, y_pred):
    if y_pred.ndim == 1:
        y_pred = np.vstack((1 - y_pred, y_pred)).T
    return imcp_score(y_true, y_pred)


def _proba(model, X):
    """Return P(class=1) for any sklearn-compatible estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Fallback for SVC-like estimators (not needed if CalibratedClassifierCV used)
    return model.decision_function(X)


SCORING = {
    "imcp":        make_scorer(imcp_score_adapted, response_method="predict_proba"),
    "roc_auc":     make_scorer(roc_auc_score, response_method="predict_proba"),
    "pr_auc":      make_scorer(average_precision_score, response_method="predict_proba"),
    "f1":          make_scorer(f1_score, pos_label=1),
    "precision":   make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall":      make_scorer(recall_score, pos_label=1, zero_division=0),
    "balanced_acc": make_scorer(balanced_accuracy_score),
    "mcc":         make_scorer(matthews_corrcoef),
    "kappa":       make_scorer(cohen_kappa_score),
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_models(models_dir, splits_path, output_dir, cv_folds=5):
    
    os.makedirs(output_dir, exist_ok=True)
    splits = np.load(splits_path, allow_pickle=True).item()

    model_files = sorted(glob(os.path.join(models_dir, "*.pkl")))
    if not model_files:
        raise FileNotFoundError(f"No .pkl models found in {models_dir}")

    sk = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    all_results = {}

    for model_path in model_files:
        model_path = Path(model_path)
        fname = model_path.stem  # e.g. "STRAIN_lgbm_by_undersampling"
        parts = fname.split("_")
        strain_name = parts[0]
        model_name = parts[1] if len(parts) > 1 else "unknown"

        print(f"\n=== Evaluating {strain_name} / {model_name} ===")
        print(f"Loading: {model_path}")

        # Load the *original* (already-specified) data path from the model's training run.
        strain_files = glob(os.path.join(
            os.environ.get("STRAIN_EMBS_DIR", ""), f"{strain_name}*.npz"))
        # If you prefer, replace the line above with a hardcoded path pattern
        if not strain_files:
            print(f"[WARN] Could not find embeddings for {strain_name}; skipping.")
            continue

        data = np.load(strain_files[0])["arr_0"]
        train_idx = splits[strain_name]["train"]
        X = data[train_idx, :-1]
        y = data[train_idx, -1].astype(int)
        del data
        gc.collect()

        bs = joblib.load(model_path)          # BayesSearchCV object
        best_pipe = bs.best_estimator_

        # ---- Cross-validation over multiple metrics ----
        cv_results = cross_validate(
            best_pipe,
            X,
            y,
            cv=sk,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
            error_score=np.nan,
        )

        # Aggregate mean ± std for each metric
        summary = {}
        for metric in SCORING:
            key = f"test_{metric}"
            vals = cv_results[key]
            summary[metric] = {
                "mean": float(np.nanmean(vals)),
                "std":  float(np.nanstd(vals)),
                "folds": [float(v) for v in vals],
            }

        # Also log the tuned hyperparameters
        best_params = {k: (v.__class__.__name__ if not isinstance(v, (int, float, str, bool, type(None)))
                           else v)
                       for k, v in bs.best_params_.items()}

        all_results[f"{strain_name}_{model_name}"] = {
            "strain": strain_name,
            "model": model_name,
            "best_params": best_params,
            "cv_metrics": summary,
        }

        # Pretty print
        for m, s in summary.items():
            print(f"  {m:14s}: {s['mean']:.4f} ± {s['std']:.4f}")

        del X, y, bs, best_pipe
        gc.collect()

    # Save everything
    out_file = os.path.join(output_dir, "cv_multi_metric_results.pkl")
    joblib.dump(all_results, out_file)
    print(f"\nSaved aggregated results to {out_file}")

    # Also dump a flat CSV for easy reporting
    try:
        import pandas as pd
        rows = []
        for key, r in all_results.items():
            for metric, s in r["cv_metrics"].items():
                rows.append({
                    "strain": r["strain"],
                    "model": r["model"],
                    "metric": metric,
                    "mean": s["mean"],
                    "std": s["std"],
                })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "cv_multi_metric_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV summary to {csv_path}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved best models with multiple metrics via CV."
    )
    parser.add_argument("--models_dir", type=str, required=True,
                        help="Directory containing *_<model>_by_undersampling.pkl files")
    parser.add_argument("--splits_path", type=str, required=True,
                        help="Path to the .npy splits file used during training")
    parser.add_argument("--strain_embs_dir", type=str, required=True,
                        help="Directory containing the strain .npz embedding files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    # Make the embeddings dir available to the worker
    os.environ["STRAIN_EMBS_DIR"] = args.strain_embs_dir

    evaluate_models(
        models_dir=args.models_dir,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        cv_folds=args.cv,
    )


if __name__ == "__main__":
    main()