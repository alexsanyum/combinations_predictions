from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

def load_data(comb_data_path, norm_small_mols_path, small_mol_embs_path, norm_abx_path, abx_embs_path):

    # Load filtered combination data
    valid_combs = pd.read_csv(comb_data_path)
    
    # Load normalized small molecule and antibiotic data along with their embeddings
    norm_small_mols = pd.read_csv(norm_small_mols_path)
    norm_abx = pd.read_csv(norm_abx_path)
    small_mol_embs = np.load(small_mol_embs_path)
    abx_embs = np.load(abx_embs_path)
    
    return valid_combs, norm_small_mols, small_mol_embs, norm_abx, abx_embs

def build_map_indexes(norm_abx, norm_small_mols):

    # Create mapping from antibiotic and small molecule names to their respective embedding indices
    abx_to_index = pd.Series(norm_abx.index, index=norm_abx['abx_name']).to_dict()
    small_mol_to_index = pd.Series(norm_small_mols.index, index=norm_small_mols['cp_name']).to_dict()
    return abx_to_index, small_mol_to_index

def concatenate_embeddings_and_save(valid_combs, abx_to_index, small_mol_to_index, small_mol_embs, abx_embs, output_dir):
    # Split data by strain_name
    strains = valid_combs['strain_name'].unique()
    
    for strain in strains:
        strain_data = valid_combs[valid_combs['strain_name'] == strain]

        # Save also strain data with just ID for reference
        strain_data[['abx_name', 'cp_name', 'bliss_med']].to_csv(os.path.join(output_dir, f"{strain}_data.csv"), index=False)
        
        # Map ids to their respective embedding indices
        abx_indices = strain_data['abx_name'].map(abx_to_index).values
        small_mol_indices = strain_data['cp_name'].map(small_mol_to_index).values

        # Broadcast embeddings to match the number of combinations
        abx_embeddings = abx_embs['embeddings'][abx_indices]
        small_mol_embeddings = small_mol_embs['embeddings'][small_mol_indices]
        bliss_med = strain_data['bliss_med'].values.reshape(-1, 1)

        # Convert bliss med to labels (1: abs(bliss_med > 0.3), else 0)

        def bliss_map(bliss_med):
            return 0 if np.abs(bliss_med) < 0.3 else 1
        
        bliss_labels = np.array([bliss_map(x) for x in bliss_med])

        # Builc [abx_embeddings, small_mol_embeddings, bliss_med] for each combination
        combined_embeddings = np.concatenate((abx_embeddings, small_mol_embeddings), axis=1)
        combined_embeddings = np.concatenate((combined_embeddings, bliss_labels.reshape(-1, 1)), axis=1)

        # Save compressed embeddings for the strain
        print(f"Saving embeddings for strain: {strain} with shape {combined_embeddings.shape}")
        np.savez_compressed(os.path.join(output_dir, f"{strain}_embeddings.npz"), comb_embs=combined_embeddings)

        # Free up memory
        del abx_embeddings, small_mol_embeddings, bliss_med, combined_embeddings
    print(f"Embeddings for all strains have been saved in {output_dir}.")
    return

def main():
    parser = ArgumentParser(description="Concatenate embeddings for each strain and save them.")
    parser.add_argument("--comb_data", required=True, help="Path to the filtered combination data CSV.")
    parser.add_argument("--norm_small_mols", required=True, help="Path to the normalized small molecules CSV.")
    parser.add_argument("--small_mol_embs", required=True, help="Path to the small molecule embeddings NPZ file.")
    parser.add_argument("--norm_abx", required=True, help="Path to the normalized antibiotics CSV.")
    parser.add_argument("--abx_embs", required=True, help="Path to the antibiotic embeddings NPZ file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the concatenated embeddings.")

    args = parser.parse_args()

    # Load data
    valid_combs, norm_small_mols, small_mol_embs, norm_abx, abx_embs = load_data(
        args.comb_data, args.norm_small_mols, args.small_mol_embs, args.norm_abx, args.abx_embs
    )

    # Build mapping indexes
    abx_to_index, small_mol_to_index = build_map_indexes(norm_abx, norm_small_mols)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Concatenate embeddings and save
    concatenate_embeddings_and_save(valid_combs, abx_to_index, small_mol_to_index, small_mol_embs, abx_embs, args.output_dir)

    return 

if __name__ == "__main__":
    main()