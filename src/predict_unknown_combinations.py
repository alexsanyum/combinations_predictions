import pandas as pd
import numpy as np
import glob
import os
import argparse
import joblib


def load_data(abx_norm_path, sm_norm_path, abx_embs_path, sm_embs_path):
    # Load normalized antibiotic and small molecule data along with their embeddings
    abx_norm = pd.read_csv(abx_norm_path).dropna()
    sm_norm = pd.read_csv(sm_norm_path).dropna()

    with np.load(abx_embs_path) as data:
        abx_embs = data['embeddings']

    with np.load(sm_embs_path) as data:
        sm_embs = data['embeddings']

    return abx_norm, sm_norm, abx_embs, sm_embs

def build_map_indexes(abx_norm, sm_norm):
    # Create mapping from antibiotic and small molecule names to their respective embedding indices
    abx_to_index = pd.Series(abx_norm.index, index=abx_norm['abx_name']).to_dict()
    sm_to_index = pd.Series(sm_norm.index, index=sm_norm['cp_name']).to_dict()
    return abx_to_index, sm_to_index

def generate_all_combinations(abx_norm, sm_norm, output_dir):
    # Generate all possible combinations of antibiotics and small molecules if not already generated
    if not glob.glob(os.path.join(output_dir, "tmp_all_combinations.csv")):
        index = pd.MultiIndex.from_product([abx_norm['abx_name'], sm_norm['cp_name']], names=['abx_name', 'cp_name'])
        tmp_all_combinations = index.to_frame().reset_index(drop=True)
        tmp_all_combinations.to_csv(os.path.join(output_dir, "tmp_all_combinations.csv"), index=False)
    else:
        tmp_all_combinations = pd.read_csv(os.path.join(output_dir, "tmp_all_combinations.csv"))
    
    return tmp_all_combinations

def filter_unknown_combinations(strain_comb_file, tmp_all_combinations, output_dir):
    # Load existing combinations for the strain
    comb_data = pd.read_csv(strain_comb_file)
    existing_combinations = set(zip(comb_data['abx_name'], comb_data['cp_name']))

    # Fast vectorized check: zipped list evaluated against set
    all_tuples = list(zip(tmp_all_combinations['abx_name'], tmp_all_combinations['cp_name']))
    mask = [t not in existing_combinations for t in all_tuples]

    new_combinations = tmp_all_combinations[mask]
    return new_combinations

def concatenate_embeddings_for_unknown_combinations(unknown_combinations, abx_to_index, sm_to_index, abx_embs, sm_embs):
    # Map ids to their respective embedding indices
    abx_indices = unknown_combinations['abx_name'].map(abx_to_index).values
    sm_indices = unknown_combinations['cp_name'].map(sm_to_index).values

    # Broadcast embeddings to match the number of combinations
    abx_embeddings = abx_embs[abx_indices]
    sm_embeddings = sm_embs[sm_indices]

    # Concatenate embeddings for each combination
    combined_embeddings = np.concatenate([abx_embeddings, sm_embeddings], axis=1)
    return combined_embeddings

def predict_and_save(strain_name, model_file, unknown_combinations, combined_embeddings, output_dir):
    # Load the model
    model = joblib.load(model_file)

    # Make predictions
    predictions = model.predict(combined_embeddings)
    predictions_prob = model.predict_proba(combined_embeddings)[:, 1]  # Assuming binary classification and we want the probability of the positive class

    # Save predictions to CSV
    unknown_combinations['predictions'] = predictions
    unknown_combinations['predictions_prob'] = predictions_prob
    unknown_combinations.to_csv(os.path.join(output_dir, f"{strain_name}_predictions.csv"), index=False)


def main():
    arg_parser = argparse.ArgumentParser(description="Process unknown combinations for each strain.")

    arg_parser.add_argument("--abx_norm", help="Path to the normalized antibiotic data")
    arg_parser.add_argument("--sm_norm", help="Path to the normalized small molecule data")
    arg_parser.add_argument("--abx_embs", help="Path to the antibiotic embeddings")
    arg_parser.add_argument("--sm_embs", help="Path to the small molecule embeddings")
    arg_parser.add_argument("--strain_comb", help="Path to the strain combinations data. Path must contain .csv and .npz files for each strain. ")
    arg_parser.add_argument("--models_path", help="Path to the trained models")
    arg_parser.add_argument("--output_dir", help="Directory to save the output")

    args = arg_parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Load data
    print("Loading data...")
    abx_norm, sm_norm, abx_embs, sm_embs = load_data(args.abx_norm, 
                                                     args.sm_norm, 
                                                     args.abx_embs, 
                                                     args.sm_embs)
    
    print("Building mapping from names to embedding indices...")
    # Build mapping from names to embedding indices
    abx_to_index, sm_to_index = build_map_indexes(abx_norm, sm_norm)

    print("Generating all possible combinations of antibiotics and small molecules...")
    # Generate all possible combinations of antibiotics and small molecules
    tmp_all_combinations = generate_all_combinations(abx_norm, sm_norm, args.output_dir)


    comb_data = glob.glob(os.path.join(args.strain_comb, "*.csv"))

    # Each model corresponds to a strain, so we will process each strain's combinations and make predictions
    for comb_file in comb_data:
        # Identify corresponding strain and model

        strain_name = os.path.basename(comb_file).split("_")[0]
        print(f"Processing strain: {strain_name}")
        model_file = glob.glob(os.path.join(args.models_path, f"{strain_name}*.pkl"))
        if not model_file:
            print(f"No model found for strain {strain_name}. Skipping...")
            continue
        print(f"Using model: {model_file[0]}")
        # Filter unknown combinations
        print(f"Filtering unknown combinations in {comb_file}...")
        unknown_combinations = filter_unknown_combinations(comb_file, tmp_all_combinations, args.output_dir)

        print(f"Found {len(unknown_combinations)} unknown combinations for strain {strain_name}.")
        # Concatenate embeddings for unknown combinations
        combined_embeddings = concatenate_embeddings_for_unknown_combinations(unknown_combinations, abx_to_index, sm_to_index, abx_embs, sm_embs)

        # Predict and save results
        print(f"Making predictions for strain {strain_name} and saving results...")
        predict_and_save(strain_name, model_file[0], unknown_combinations, combined_embeddings, args.output_dir)
        
if __name__ == "__main__":
    main()