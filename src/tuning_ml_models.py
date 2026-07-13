import argparse
from glob import glob
from pathlib import Path
import os
import joblib
import numpy as np
import gc

# Import libraries for undersampling
from src.cluster_undersampling import ClusterUnderSampler
from sklearn.cluster import KMeans

# Import libraries for hyperparameter tuning with Bayesian optimization
from skopt.space import Real, Categorical, Integer
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

# Adapted IMCP score function, adapted version of the original imcp_score function to use trapezo
from sklearn.metrics import make_scorer
from src.imcp import imcp_score 

# Import classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def imcp_score_adapted(y_true, y_pred):
    if y_pred.ndim == 1:
        y_pred = np.vstack((1 - y_pred, y_pred)).T
    return imcp_score(y_true, y_pred)

def get_single_model_and_params(model_name):
    """Returns only the requested model and search space to avoid loading unused setups."""
    sampler = ClusterUnderSampler(random_sample_seed=42)
    sampler_params = {
        "sampler__clustering_model": Categorical([KMeans]),
        "sampler__n_clusters": Integer(2, 10, prior='uniform'),
        "sampler__reduction_percentage_class": Real(0.1, 0.5, prior='uniform')
    }

    if model_name == "lr":
        model_obj = LogisticRegression(random_state=42, class_weight='balanced', solver="saga")
        grid = {
            "lr__C": Real(1e-3, 1e3, prior="log-uniform"),
            "lr__max_iter": Integer(100, 5000, prior="uniform"),
            "lr__l1_ratio": Real(0.1, 1, prior="uniform")
        }
    elif model_name == "svc":
        base_svc = SVC(random_state=42, class_weight='balanced')
        model_obj = CalibratedClassifierCV(base_svc, ensemble=False)
        grid = {
            "svc__estimator__kernel": Categorical(["rbf", "sigmoid"]),
            "svc__estimator__degree": Integer(2, 5),
            "svc__estimator__gamma": Real(1e-3, 1e+1, prior="log-uniform"),
            "svc__estimator__C": Real(1e-1, 10, prior="log-uniform"),
            "svc__estimator__max_iter": Integer(500, 3000)
        }
    elif model_name == "rf":
        model_obj = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
        grid = {
            "rf__n_estimators": Integer(100, 300),
            "rf__max_depth": Integer(3, 15),
            "rf__min_samples_leaf": Integer(2, 10),
            "rf__criterion": Categorical(["gini", "entropy"])
        }
    elif model_name == "xgb":
        model_obj = XGBClassifier(random_state=42, n_jobs=1)
        grid = {
            "xgb__n_estimators": Integer(50, 300),
            "xgb__max_depth": Integer(1, 15),
            "xgb__learning_rate": Real(0.01, 0.3, prior='log-uniform')
        }
    elif model_name == "lgbm":
        model_obj = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=1, verbosity=-1)
        grid = {
            "lgbm__max_depth": Integer(3, 12),
            "lgbm__num_leaves": Integer(31, 100),
            "lgbm__n_estimators": Integer(100, 300), 
            "lgbm__learning_rate": Real(1e-3, 0.1, prior="log-uniform")
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    params = {**grid, **sampler_params}
    return sampler, model_obj, params


def run_pipeline(strain_embs_path, splits_path, output_models_dir, target_model, n_iter, cv_folds, n_jobs=1):
    """Executes the data loading, training loop, and model checkpoint saving."""
    os.makedirs(output_models_dir, exist_ok=True)

    dir_list = glob(strain_embs_path)
    train_test_splits = np.load(splits_path, allow_pickle=True).item()

    imcp_scorer = make_scorer(imcp_score_adapted, response_method="predict_proba")
    sampler, model_obj, params = get_single_model_and_params(target_model)

    sk = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for strain in dir_list:
        
        out_path = os.path.join(output_models_dir, f"{strain_name}_{target_model}_by_undersampling.pkl")
        if os.path.exists(out_path):
            print(f"Model for strain {strain_name} already exists at {out_path}. Skipping...")
            continue

        strain_path = Path(strain)
        strain_name = str(strain_path.name).split("_")[0]
        print(f"\nProcessing strain: {strain_name}")

        data = np.load(strain_path)['arr_0']
        
        train_indices = train_test_splits[strain_name]["train"]
    
        X_train = data[train_indices, :-1]
        y_train = data[train_indices, -1]

        del data
        gc.collect()

        pipe = Pipeline([
            ("sampler", sampler),
            (target_model, model_obj)
        ])
        bs = BayesSearchCV(
            pipe, 
            search_spaces=params, 
            scoring=imcp_scorer, 
            verbose=1, 
            error_score=np.nan, 
            n_iter=n_iter, 
            cv=sk, 
            n_jobs=n_jobs, 
            random_state=42
        )
        with joblib.parallel_backend('loky', n_jobs=n_jobs):
            bs.fit(X_train, y_train)
        
        joblib.dump(bs, out_path)
        print(f"Saved model for strain {strain_name} to {out_path}")
        del X_train, y_train, bs
        gc.collect()
            
def main():

    
    # Arguments replacing your UPPERCASE variables
    parser = argparse.ArgumentParser(description="Train models using Bayesian Optimization and Undersampling.")
    parser.add_argument("--strain_embs", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["rf", "xgb", "lgbm", "lr", "svc"], help="Specific model to run")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    run_pipeline(
        strain_embs_path=args.strain_embs,
        splits_path=args.splits_path,
        output_models_dir=args.output_dir,
        target_model=args.model,
        n_iter=args.n_iter,
        cv_folds=args.cv,
        n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()