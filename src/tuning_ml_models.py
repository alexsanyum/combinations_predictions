import argparse
from glob import glob
from pathlib import Path
import os
import joblib
import numpy as np
import gc

# Import libraries for undersampling
from src.cluster_undersampling import ClusterUnderSampler
from sklearn.cluster import KMeans, AgglomerativeClustering

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


def get_models_and_params():
    """Defines and returns the models and their respective hyperparameter search spaces."""
    # Load and set parameters for the ClusterUnderSampler
    sampler = ClusterUnderSampler(random_sample_seed=42)
    sampler_params = {
        "sampler__clustering_model": Categorical([KMeans]),
        "sampler__n_clusters": Integer(2, 10, prior='uniform'),
        "sampler__reduction_percentage_class": Real(0.1, 0.5, prior='uniform')
    }

    # Define base models
    lr = LogisticRegression(random_state=42, class_weight='balanced', solver="saga")
    base_svc = SVC(random_state=42, class_weight='balanced')
    svc = CalibratedClassifierCV(base_svc, ensemble=False)
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
    lgbm = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=1) 
    xgb = XGBClassifier(random_state=42, n_jobs=1) 

    # Parameter spaces
    lr_params = {
        "lr__C": Real(1e-3, 1e3, prior="log-uniform"),
        "lr__max_iter": Integer(100, 10000, prior="uniform"),
        "lr__l1_ratio": Real(0.1, 1, prior="uniform")
    }

    svc_params = {
        "svc__estimator__kernel": Categorical(["rbf", "sigmoid"]),
        "svc__estimator__degree": Integer(2, 7),
        "svc__estimator__gamma": Real(1e-3, 1e+1, prior="log-uniform"),
        "svc__estimator__C": Real(1e-1, 10, prior="log-uniform"),
        "svc__estimator__max_iter": Integer(500, 5000)
    }

    rf_params = {
        "rf__n_estimators": Integer(100, 500),
        "rf__max_depth": Integer(3, 15),
        "rf__min_samples_leaf": Integer(2, 10),
        "rf__min_samples_split": Integer(2, 10),
        "rf__criterion": Categorical(["gini", "entropy", "log_loss"]),
        "rf__class_weight": Categorical([None, "balanced", "balanced_subsample"])
    }

    xgb_params = {
        "xgb__n_estimators": Integer(50, 500),
        "xgb__max_depth": Integer(1, 20),
        "xgb__learning_rate": Real(0.01, 0.3, prior='log-uniform')
    }

    lgbm_params = {
        "lgbm__max_depth": Integer(3, 12),
        "lgbm__boosting_type": Categorical(["gbdt", "dart"]), 
        "lgbm__num_leaves": Integer(31, 100),
        "lgbm__min_child_samples": Integer(20, 70), 
        "lgbm__n_estimators": Integer(100, 500), 
        "lgbm__learning_rate": Real(1e-3, 0.1, prior="log-uniform"), 
        "lgbm__lambda_l2": Real(1e-5, 0.1, prior="log-uniform"),
        "lgbm__feature_fraction": Real(0.5, 1.0)
    }

    models_params = {
        "rf": (rf, rf_params),
        "xgb": (xgb, xgb_params),
        "lgbm": (lgbm, lgbm_params),
        "lr": (lr, lr_params),
        "svc": (svc, svc_params)
    }
    
    return sampler, sampler_params, models_params


def run_pipeline(strain_embs_path, splits_path, output_models_dir, n_iter, cv_folds, n_jobs=1):
    """Executes the data loading, training loop, and model checkpoint saving."""
    os.makedirs(output_models_dir, exist_ok=True)

    dir_list = glob(strain_embs_path)
    train_test_splits = np.load(splits_path, allow_pickle=True).item()

    imcp_scorer = make_scorer(imcp_score_adapted, response_method="predict_proba")
    sampler, sampler_params, models_params = get_models_and_params()

    sk = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for strain in dir_list:
        strain_path = Path(strain)
        strain_name = str(strain_path.name).split("_")[0]
        print(f"\nProcessing strain: {strain_name}")

        data = np.load(strain_path)['comb_embs']
        
        train_indices = train_test_splits[strain_name]["train"]
    
        X_train = data[train_indices, :-1]
        y_train = data[train_indices, -1]

        del data
        gc.collect()

        for model_name, (model_obj, model_grid) in models_params.items():
            print(f"  Optimizing {model_name}...")
            params = {**model_grid, **sampler_params}
            pipe = Pipeline([
                ("sampler", sampler),
                (model_name, model_obj)
            ])

            bs = BayesSearchCV(
                pipe, 
                search_spaces=params, 
                scoring=imcp_scorer, 
                verbose=2, 
                error_score=np.nan, 
                n_iter=n_iter, 
                cv=sk, 
                n_jobs=n_jobs, 
                random_state=42
            )
            
            bs.fit(X_train, y_train)

            output_name = os.path.join(output_models_dir, f"{strain_name}_{model_name}_bayes_undersampling.pkl")
            joblib.dump(bs, output_name)
            print(f"  Saved: {output_name}")
            
def main():
    parser = argparse.ArgumentParser(description="Train models using Bayesian Optimization and Undersampling.")
    
    # Arguments replacing your UPPERCASE variables
    parser.add_argument("--strain_embs", type=str, required=True, help="Path pattern to strain embeddings (e.g. 'data/strains_embs/*.npz')")
    parser.add_argument("--splits_path", type=str, required=True, help="Path to train/test splits .npy file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory path to save output models")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for BayesSearchCV (default: 1)")
    # The two requested flags (with default fallbacks matching your original code)
    parser.add_argument("--n_iter", type=int, default=2, help="Number of iterations for BayesSearchCV (default: 2)")
    parser.add_argument("--cv", type=int, default=2, help="Number of cross-validation folds for BayesSearchCV (default: 2)")

    args = parser.parse_args()

    run_pipeline(
        strain_embs_path=args.strain_embs,
        splits_path=args.splits_path,
        output_models_dir=args.output_dir,
        n_iter=args.n_iter,
        cv_folds=args.cv,
        n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()