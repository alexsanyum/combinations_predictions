import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from src.cluster_undersampling import ClusterUnderSampler
from sklearn.cluster import KMeans
import argparse


def read_embs(path_to_embs):
    with np.load(path_to_embs) as data:
        data = data['arr_0']
        X = data[:, :-1]
        y = data[:, -1]
    return X,y

def run_one(X,y,params,seed, out_dir):
    sampler = ClusterUnderSampler(random_sample_seed=seed, **params)
    sampler.fit_resample(X,y)
    idx = sampler.under_sample_index_

    # Save the undersample index to a .npy file
    fname = os.path.join(out_dir, 
                         f"sampling_{params['clustering_model'].__name__}_k_{params['n_clusters']}_rc_{params['reduction_percentage_class']}_{seed}.npy")
    np.save(fname, idx)

    return {'n_clusters': params['n_clusters'], 
            'clustering_model': params['clustering_model'].__name__,
            'reduction_percentage_class': params['reduction_percentage_class'], 
            'seed': seed, 
            'path': fname}

def evaluate_dataset(X,y,params_grid,seeds, out_dir, n_jobs=-1):
    taks = [
        (params,seed) for params in params_grid for seed in seeds
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(X,y,params,seed, out_dir) for params,seed in taks
    )
    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser(description="Iterate undersampling on a dataset varying the parameters.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the .npz file containing the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the results DataFrame as a .csv file.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs to run. Default is -1 (use all available cores).")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for each parameter combination. Default is 5.")
    parser.add_argument("--test_mode", action='store_true', help="If set, runs in test mode with a smaller dataset.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.data_path))[0]
    base_name = base_name.split('_embs')[0]
    idx_dir = os.path.join(args.output_dir, f"{base_name}_indices")
    os.makedirs(idx_dir, exist_ok=True)
    output_name = os.path.join(args.output_dir, f"{base_name}_undersampling_iterations_idx.csv")

    SEEDS = list(range(args.iterations))

    # Read the dataset
    X, y = read_embs(args.data_path)

    # Define the parameter grid for undersampling
    sampler_parameters = {"n_clusters":[2,3,4,5,6,7,8,9,10],
                         "clustering_model":[KMeans],
                         "reduction_percentage_class":[ 0.1, 0.25, 0.5, 0.75]}

    if args.test_mode:
        print("Running in test mode: using a smaller dataset for quick testing.")
        X = X[:1000]
        y = y[:1000]
        sampler_parameters = {"n_clusters":[2,3],
                             "clustering_model":[KMeans],
                             "reduction_percentage_class":[ 0.1, 0.5]}
        
    params_grid = ParameterGrid(sampler_parameters)

    df_results = evaluate_dataset(X, y, params_grid, SEEDS, idx_dir,n_jobs=args.n_jobs)

   # Save the results DataFrame to a CSV file
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.data_path))[0]
    output_name = os.path.join(args.output_dir, f"{base_name}_undersampling_iterations_idx.csv")
 
    df_results.to_csv(output_name, index=False)

if __name__ == "__main__":
    '''Usage
    python src/undersampling_iterations.py --data_path data/strains_embs/Ab17978_embs.npz --output_dir data/undersampling_iterations_results/ --n_jobs 32 --iterations 1000
    '''
    main()