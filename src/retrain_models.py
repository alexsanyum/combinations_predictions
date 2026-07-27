import argparse
import gc
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Local imports
from src.tuning_ml_models import imcp_score_adapted


def evaluate_model(model, X_test, y_test):
    """Calculates classification metrics for a trained model on a test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "imcp": imcp_score_adapted(y_test, y_pred),
        "fbeta": fbeta_score(y_test, y_pred, beta=2),
    }, y_proba


def retrain_and_evaluate_strain(
    strain_name, model_path, embs_path, train_indices, test_indices, output_dir
):
    """Retrains the best estimator pipeline on full train set and evaluates on test set."""
    print(f"\n--- Processing strain: {strain_name} ---")

    # Load search model
    if not os.path.exists(model_path):
        print(f"Skipping {strain_name}: Model checkpoint not found at {model_path}")
        return None

    search_result = joblib.load(model_path)
    best_pipeline = search_result.best_estimator_

    # Load dataset matrix
    data = np.load(embs_path)

    matrix = data['arr_0']

    X_train, y_train = matrix[train_indices, :-1], matrix[train_indices, -1]
    X_test, y_test = matrix[test_indices, :-1], matrix[test_indices, -1]

    # Free raw array memory
    del data, matrix
    gc.collect()

    # Retrain pipeline on full training set
    print(f"Retraining full pipeline for {strain_name}...")
    best_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    print(f"Evaluating {strain_name} on test set...")
    metrics, y_proba = evaluate_model(best_pipeline, X_test, y_test)

    metrics["strain"] = strain_name

    out_model_path = os.path.join(output_dir, f"{strain_name}_final_model.pkl")
    joblib.dump(best_pipeline.steps[-1][1], out_model_path)
    print(f"Saved retrained model to: {out_model_path}")

    # Save predictions
    out_preds_path = os.path.join(output_dir, f"{strain_name}_predictions.csv")
    preds_df = pd.DataFrame({
        "true_labels": y_test,
        "predictions_prob": y_proba})
    preds_df.to_csv(out_preds_path, index=False)
    print(f"Saved predictions to: {out_preds_path}")
    # Free memory
    del X_train, y_train, X_test, y_test
    gc.collect()

    return metrics


def run_evaluation_pipeline(models_dir, embs_dir, splits_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_test_splits = np.load(splits_path, allow_pickle=True).item()

    metrics_list = []
    models_paths = os.listdir(models_dir)

    embs_paths = os.listdir(embs_dir)
    # embs files ends with "_embs.npz"
    embs_paths = [f for f in embs_paths if f.endswith("_embeddings.npz")]

    for strain in train_test_splits.keys():
        # find the model and embedding file for the strain
        model_file = [f for f in models_paths if f.startswith(strain) and f.endswith(".pkl")][0]
        embs_file = [f for f in embs_paths if f.startswith(strain) and f.endswith("_embeddings.npz")][0]
        
        train_idx = train_test_splits[strain]["train"]
        test_idx = train_test_splits[strain]["test"]
    
        metrics = retrain_and_evaluate_strain(
            strain,
            os.path.join(models_dir, model_file),
            os.path.join(embs_dir, embs_file),
            train_idx,
            test_idx,
            output_dir
        )

        if metrics is not None:
            metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(output_dir, "retrained_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    return metrics_list


def main():
    parser = argparse.ArgumentParser(description="Retrain and evaluate models for each strain.")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing trained model Bayes Search objects.")
    parser.add_argument("--embs_dir", type=str, required=True, help="Directory containing embedding matrices for each strain.")
    parser.add_argument("--splits_path", type=str, required=True, help="Path to the .npy file containing train/test splits.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save retrained models and evaluation metrics.")
    args = parser.parse_args()

    run_evaluation_pipeline(args.models_dir, args.embs_dir, args.splits_path, args.output_dir)


if __name__ == "__main__":
    main()