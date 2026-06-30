import pandas as pd
from chem_utilities import normalize_smiles_df
from argparse import ArgumentParser
import os
def load_data(path):
    return pd.read_csv(path)

def quality_bac_filter(df):
    # Filter by c_adj_batch_bliss_pval
    df_filtered = df[df['c_adj_batch_bliss_pval'] < 0.05]

    # Exclude rows with abx_name == BAC
    df_filtered = df_filtered[df_filtered['abx_name'] != 'BAC']

    # Return next columns strain_name, abx_name, cp_name, bliss_med, cp_SMILES
    return df_filtered[['strain_name', 'abx_name', 'cp_name', 'bliss_med', 'cp_SMILES']]

def process_valid_SMILES_comb(comb_data_df, abx_data_df):
    small_mol_df = comb_data_df[['cp_name', 'cp_SMILES']].drop_duplicates()

    abx_df = comb_data_df[['abx_name']].drop_duplicates()
    abx_smiles_df = abx_df.merge(abx_data_df[['abx_name', 'abx_SMILES']], on='abx_name', how='left')

    return small_mol_df, abx_smiles_df

def normalize_smiles(smiles_df, col_name):
    smiles_df = normalize_smiles_df(smiles_df, col_name, protonate=False)
    return smiles_df

def identify_invalid_smiles(small_mol_df, abx_smiles_df):
    # Identify invalid SMILES for small molecules
    invalid_smiles_sm = small_mol_df[small_mol_df['cp_SMILES'].isnull()]

    # Identify invalid SMILES for antibiotics
    invalid_smiles_abx = abx_smiles_df[abx_smiles_df['abx_SMILES'].isnull()]

    return invalid_smiles_sm, invalid_smiles_abx

def filter_valid_smiles(comb_data_df, invalid_smiles_sm, invalid_smiles_abx):
    # Drop rows with invalid SMILES
    valid_comb_data = comb_data_df[
        ~comb_data_df['abx_name'].isin(invalid_smiles_abx['abx_name'])
        & ~comb_data_df['cp_name'].isin(invalid_smiles_sm['cp_name'])
    ]

    return valid_comb_data

def main():
    parser = ArgumentParser()
    parser.add_argument('--comb_data_path', type=str, required=True, help='Path to the combination data CSV file')
    parser.add_argument('--abx_data_path', type=str, required=True, help='Path to the antibiotic data CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the valid combination data CSV file')
    args = parser.parse_args()

    data = load_data(args.comb_data_path)
    abx_data = load_data(args.abx_data_path)

    # Quality and BAC filtering
    filtered_data = quality_bac_filter(data)

    # Extract small molecules and antibiotics with their SMILES
    small_mol_df, abx_smiles_df = process_valid_SMILES_comb(filtered_data, abx_data)
    
    
    # Normalize SMILES
    print("Normalizing and saving SMILES...")
    abx_smiles_df = normalize_smiles(abx_smiles_df, 'abx_SMILES')
    abx_smiles_df.to_csv(os.path.join(os.path.dirname(args.output_path), 'normalized_abx.csv'), index=False)
    
    small_mol_df = normalize_smiles(small_mol_df, 'cp_SMILES')
    small_mol_df.to_csv(os.path.join(os.path.dirname(args.output_path), 'normalized_small_mol.csv'), index=False)

    # Identify invalid SMILES
    invalid_smiles_sm, invalid_smiles_abx = identify_invalid_smiles(small_mol_df, abx_smiles_df)

    # Filter valid combination data
    print("Filtering valid combination data...")
    valid_comb_data = filter_valid_smiles(filtered_data, invalid_smiles_sm, invalid_smiles_abx)

    # Save data to CSV
    print(f"Saving normalized SMILES and valid combination data to {args.output_path}...")
    valid_comb_data.to_csv(os.path.join(os.path.dirname(args.output_path), 'valid_comb_data.csv'), index=False)


    return

if __name__ == "__main__":
    main()