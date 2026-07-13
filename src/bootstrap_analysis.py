import os
import argparse
import joblib
import numpy as np
import pandas as pd
from glob import glob
from sklearn.utils import resample
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, log_loss

# Local imports
from src.tuning_ml_models import imcp_score_adapted
from src.cluster_undersampling import ClusterUnderSampler

# --- Helper to Dynamically Instantiate Base Models ---
def get_base_model(model_name):
    """Returns the unconfigured classifier class based on model name."""
    model_name = model_name.lower()
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=42, verbosity=-1)
    elif model_name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=42, eval_metric="logloss")
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)
    elif model_name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == "svc":
        from sklearn.svm import SVC
        return SVC(random_state=42, probability=True)
    else:
        raise ValueError(f"Model type '{model_name}' is not recognized.")

# --- Paths Resolution Function ---
def get_paths_dictionary(path_to_models, path_to_data, path_to_indices, strain_name, model_name):
    """Loads indices and resolves paths specifically for the target strain."""
    indices_dict = np.load(path_to_indices, allow_pickle=True).item()
    
    if strain_name not in indices_dict:
        raise KeyError(f"Strain '{strain_name}' not found in the split indices file.")
        
    # Search for files dynamically matching naming convention
    model_pattern = os.path.join(path_to_models, f"{strain_name}_{model_name}_*.pkl")
    data_pattern = os.path.join(path_to_data, f"{strain_name}_embeddings.npz")
    
    model_files = glob(model_pattern)
    data_files = glob(data_pattern)
    
    if not model_files:
        # Fallback to check if underscore formatting is slightly different
        model_pattern_fallback = os.path.join(path_to_models, f"{strain_name}*.pkl")
        model_files = [f for f in glob(model_pattern_fallback) if model_name in os.path.basename(f)]
        
    if not model_files:
        raise FileNotFoundError(f"Could not find any trained .pkl model matching: {model_pattern}")
    if not data_files:
        raise FileNotFoundError(f"Could not find matching .npz embeddings file for: {strain_name}")
        
    return {
        "model_path": model_files[0],
        "data_path": data_files[0],
        "indices": indices_dict[strain_name]
    }

# --- Main Bootstrap Execution Loop ---
def bootstrap_train(model, model_step_name, X_train, y_train, X_test, y_test, n_bootstraps=1000):
    # Store predictions
    predictions = [y_test]
    predictions_proba = [y_test]

    # Store metrics
    metrics = {'accuracy': [], 'roc_auc': [], 'mcc': [], 'log_loss': [], "imcp": []}
    feature_importances = []
    
    for i in range(n_bootstraps):
        print(f"Bootstrap iteration {i+1}/{n_bootstraps}...")
        
        # Resample training data (Stratified)
        X_resampled, y_resampled = resample(
            X_train, y_train, 
            replace=True, 
            random_state=42 + i, 
            stratify=y_train
        )
        
        # Fit pipeline
        model.fit(X_resampled, y_resampled)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        predictions.append(y_pred)
        predictions_proba.append(y_pred_proba)

        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
        metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
        metrics['log_loss'].append(log_loss(y_test, y_pred_proba))
        metrics['imcp'].append(imcp_score_adapted(y_test, y_pred_proba))
        
        # Extract weights/importances dynamically from the estimator step
        estimator = model.named_steps[model_step_name]
        
        # Check if the estimator is calibrated
        if hasattr(estimator, 'calibrated_classifiers_'):
            # Grab average weights across calibrated folds if using CalibratedClassifierCV
            sub_estimators = [c.estimator for c in estimator.calibrated_classifiers_]
            if hasattr(sub_estimators[0], 'coef_'):
                coefs = np.mean([sub.coef_[0] for sub in sub_estimators], axis=0)
                feature_importances.append(coefs)
            elif hasattr(sub_estimators[0], 'feature_importances_'):
                importances = np.mean([sub.feature_importances_ for sub in sub_estimators], axis=0)
                feature_importances.append(importances)
            else:
                feature_importances.append(None)
        else:
            if hasattr(estimator, 'feature_importances_'):
                feature_importances.append(estimator.feature_importances_)
            elif hasattr(estimator, 'coef_'):
                feature_importances.append(estimator.coef_[0])
            else:
                feature_importances.append(None) 

    return {
        'metrics': metrics, 
        'predictions': predictions, 
        'predictions_proba': predictions_proba, 
        'feature_importances': feature_importances
    }

# --- Orchestrator ---
def main():
    parser = argparse.ArgumentParser(description="Parallel Bootstrap Analysis Module")
    
    # Uppercase environment variables exposed as arguments
    parser.add_argument("--PATH_TO_MODELS", type=str, default="data/models/", help="Path to production trained models")
    parser.add_argument("--PATH_TO_DATA", type=str, default="data/strains_embs/", help="Path to directory containing strain .npz files")
    parser.add_argument("--PATH_TO_INDICES", type=str, default="data/strains_embs/train_test_splits_indices.npy", help="Path to splits .npy file")
    parser.add_argument("--OUTPUT_DIR", type=str, default="data/bootstrap_results/", help="Output results directory")
    
    # Target Task Arguments
    parser.add_argument("--STRAIN", type=str, required=True, choices=['Ab17978', 'AbLac4', 'Kp0087', 'Kp43816', 'PAO1', 'Pa0095'], help="Strain target name")
    parser.add_argument("--MODEL_NAME", type=str, required=True, choices=['lgbm', 'xgb', 'rf', 'lr', 'svc'], help="Pipeline classifier framework")
    parser.add_argument("--N_BOOTSTRAPS", type=int, default=1000, help="Total bootstrap resampling iterations")
    
    args = parser.parse_args()
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    
    print(f"Resolving assets for: {args.STRAIN} ({args.MODEL_NAME})")
    paths = get_paths_dictionary(
        path_to_models=args.PATH_TO_MODELS, 
        path_to_data=args.PATH_TO_DATA, 
        path_to_indices=args.PATH_TO_INDICES, 
        strain_name=args.STRAIN,
        model_name=args.MODEL_NAME
    )
    
    # Load optimization payload
    print(f"Loading best parameters from: {paths['model_path']}")
    bayes_search_result = joblib.load(paths['model_path'])
    
    # Get hyperparameters
    best_params = bayes_search_result.best_estimator_.get_params()
    
    # Load dataset
    data = np.load(paths['data_path'])['arr_0']
    indices = paths['indices']

    X_train, y_train = data[indices['train'], :-1], data[indices['train'], -1]
    X_test, y_test = data[indices['test'], :-1], data[indices['test'], -1]
    
    # Dynamically build baseline Pipeline using model_name as key
    base_clf = get_base_model(args.MODEL_NAME)
    pipe = Pipeline([
        ('undersampler', ClusterUnderSampler(n_clusters=5, random_state=42)),
        (args.MODEL_NAME, base_clf)
    ])
    
    # Apply parameters
    pipe = pipe.set_params(**best_params)
    
    # Override n_jobs to -1 dynamically for tree-based models if supported
    if hasattr(base_clf, 'n_jobs'):
        pipe.set_params(**{f"{args.MODEL_NAME}__n_jobs": -1})

    print(f"Executing Bootstrap runs on {args.STRAIN} (n_iterations={args.N_BOOTSTRAPS})")
    bootstrap_results = bootstrap_train(
        model=pipe, 
        model_step_name=args.MODEL_NAME, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        n_bootstraps=args.N_BOOTSTRAPS
    )
    
    # Save output cleanly
    output_filename = os.path.join(args.OUTPUT_DIR, f"bootstrap_results_{args.STRAIN}_{args.MODEL_NAME}.npz")
    np.savez(output_filename, **bootstrap_results)
    print(f"Completed! Results successfully saved to: {output_filename}")

if __name__ == "__main__":
    main()