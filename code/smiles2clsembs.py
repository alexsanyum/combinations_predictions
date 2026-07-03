from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name):
    """
    Load the pre-trained model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the pre-trained model.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    model = model.to(torch.device(device))
    return model, tokenizer

def process_smiles(file_path, column):
    smiles_df = pd.read_csv(file_path)
    
    # Create a categorical column for the SMILES strings
    smiles_df["smiles_category"] = pd.Categorical(smiles_df[column])

    # Extract the unique SMILES strings and their corresponding indices
    unique_smiles_list = smiles_df["smiles_category"].cat.categories.tolist()

    inverse_mapping = smiles_df["smiles_category"].cat.codes.values

    original_length = len(smiles_df)
    return unique_smiles_list, inverse_mapping, original_length


def generate_embeddings(unique_smiles_list, model, tokenizer, device, batch_size=32):
    
    if batch_size > 32:
            print("\n" + "="*80)
            print("WARNING: Batch size > 32 detected.")
            print("Large batch sizes alter the GPU parallel execution paths, introducing minor")
            print("floating-point numerical drift (order of 1e-6) due to hardware non-determinism.")
            print("While small, this drift affects highly precise near-zero embedding features.")
            print("\nRECOMMENDATION: Use a batch size of 32 or lower to guarantee strict")
            print("reproducibility across different runs and hardware environments.")
            print("="*80 + "\n")
            
    unique_embeddings = []
    model.eval()  # Set the model to evaluation mode

    with torch.inference_mode():  # Disable gradient calculation for inference
        for i in range(0, len(unique_smiles_list), batch_size):
            # Control for the last batch which might be smaller than batch_size
            lower_bound = i
            upper_bound = min(i + batch_size, len(unique_smiles_list))
            batch_smiles = unique_smiles_list[lower_bound:upper_bound]
            token = tokenizer(batch_smiles, 
                              padding=True, 
                              truncation=True, 
                              return_tensors="pt",
                              max_length=512).to(torch.device(device))
            
            outputs = model(**token)

            # Pull to CPU
            batch_cls = outputs.last_hidden_state.cpu().numpy()[:, 0, :]  # Use mean pooling to get a single vector representation for each SMILES
            unique_embeddings.append(batch_cls)

            # Explicitly free GPU memory after processing each batch
            del token, outputs, batch_cls
            if i % (batch_size * 10) == 0:  # Print progress every 10 batches
                torch.cuda.empty_cache()  # Clear cache to free up memory
    # stack the list of arrays into a single array
    unique_embeddings_array = np.vstack(unique_embeddings)
    
    return unique_embeddings_array

def process_embeddings(unique_embeddings_array, inverse_mapping, original_length):
    valid_mask = inverse_mapping != -1
    embeddings = np.zeros((original_length, unique_embeddings_array.shape[1]), dtype=np.float32)
    embeddings[valid_mask] = unique_embeddings_array[inverse_mapping[valid_mask]]

    return embeddings

def test(embeddings, smiles_df, column, len_test, model, tokenizer, device):
    random_indices = np.random.choice(len(smiles_df), size=len_test, replace=False)

    smiles_from_original = smiles_df.iloc[random_indices]['normalized_smiles'].notnull()
    
    # Filter out any indices where the SMILES string is null
    valid_random_indices = random_indices[smiles_from_original]

    smiles_to_test = smiles_df.iloc[valid_random_indices][column].values.tolist()


    # Filter 
    embs_from_original = generate_embeddings(smiles_to_test, model, tokenizer, device, batch_size=32)

    embs_from_final = embeddings[valid_random_indices]

    # Check if the embeddings are close enough (allowing for minor numerical differences)
    are_embeddings_close = np.allclose(embs_from_original, embs_from_final, atol=1e-6)
    if are_embeddings_close:
        print(f"Test passed: The embeddings for {len_test} random SMILES are consistent.")
    else:
        print(f"Test failed: The embeddings for {len_test} random SMILES are NOT consistent.")
    return

def main():
    parser = ArgumentParser()
    parser.add_argument('--smiles_path', type=str, required=True, help='Path to the SMILES CSV file')
    parser.add_argument('--smiles_column', type=str, required=True, help='Column name containing SMILES strings')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the embeddings CSV file')
    parser.add_argument('--len_test', type=int, default=0, help='Number of random samples for testing embeddings')
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model and tokenizer for {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Process SMILES
    print(f"Processing unique SMILES from {args.smiles_path}...")
    unique_smiles_list, inverse_mapping, original_length = process_smiles(args.smiles_path, args.smiles_column)

    # Generate embeddings
    print(f"Computing embeddings for {len(unique_smiles_list)} unique SMILES using {args.model_name}...")
    unique_embeddings_array = generate_embeddings(unique_smiles_list, model, tokenizer, device)

    # Process embeddings
    print(f"Mapping embeddings back to original SMILES order...")
    embeddings = process_embeddings(unique_embeddings_array, inverse_mapping, original_length)

    # Check if shape of original embeddings matches the expected shape
    expected_shape = (original_length, unique_embeddings_array.shape[1])
    if embeddings.shape != expected_shape:
        raise ValueError(f"Shape mismatch: Expected {expected_shape}, but got {embeddings.shape}")
    # Test embeddings

    if args.len_test > 0:
        print(f"Testing embeddings for {args.len_test} random SMILES...")
        smiles_df = pd.read_csv(args.smiles_path)
        test(embeddings, smiles_df, args.smiles_column, args.len_test, model, tokenizer, device)
    
    # Save embeddings as compressed npz file
    print(f"Saving embeddings to {args.output_path}...")
    np.savez_compressed(args.output_path, embeddings=embeddings)



if __name__ == "__main__":
    '''
    Usage
    python smiles2clsembs.py --smiles_path path/to/smiles.csv --smiles_column normalized_smiles --model_name model_name --output_path path/to/output_embeddings.csv --len_test 100
    '''
    main()