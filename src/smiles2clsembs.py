from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name):
    """Load pre-trained model and tokenizer to device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, tokenizer

def process_smiles(file_path, column):
    """Extract unique SMILES, inverse mapping, and dataset length."""
    smiles_df = pd.read_csv(file_path)
    smiles_df["smiles_category"] = pd.Categorical(smiles_df[column])
    
    unique_smiles_list = smiles_df["smiles_category"].cat.categories.tolist()
    inverse_mapping = smiles_df["smiles_category"].cat.codes.values
    return unique_smiles_list, inverse_mapping, len(smiles_df)

def generate_embeddings(unique_smiles_list, model, tokenizer, device, batch_size=32):
    """Generate CLS embeddings for a list of SMILES strings."""
    if batch_size > 32:
        print("Warning: batch_size > 32 may introduce minor numerical drift.")
            
    unique_embeddings = []
    model.eval()

    with torch.inference_mode():
        for i in range(0, len(unique_smiles_list), batch_size):
            lower_bound = i
            upper_bound = min(i + batch_size, len(unique_smiles_list))
            batch_smiles = unique_smiles_list[lower_bound:upper_bound]
            token = tokenizer(
                batch_smiles, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(device)
            
            outputs = model(**token)
            batch_cls = outputs.last_hidden_state.cpu().numpy()[:, 0, :]
            unique_embeddings.append(batch_cls)

            del token, outputs, batch_cls
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    return np.vstack(unique_embeddings)

def process_embeddings(unique_embeddings_array, inverse_mapping, original_length):
    """Map unique embeddings back to original dataset dimensions."""
    valid_mask = inverse_mapping != -1
    embeddings = np.zeros((original_length, unique_embeddings_array.shape[1]), dtype=np.float32)
    embeddings[valid_mask] = unique_embeddings_array[inverse_mapping[valid_mask]]
    return embeddings

def test(embeddings, smiles_df, column, len_test, model, tokenizer, device):
    """Validate consistency by regenerating embeddings for random samples."""
    df_clean = smiles_df.dropna(subset=[column])
    sample_df = df_clean.sample(n=min(len_test, len(df_clean)))
    
    test_smiles = sample_df[column].tolist()
    embs_from_original = generate_embeddings(test_smiles, model, tokenizer, device, batch_size=32)
    embs_from_final = embeddings[sample_df.index]

    if np.allclose(embs_from_original, embs_from_final, atol=1e-6):
        print(f"Test passed: Embeddings for {len(test_smiles)} samples match.")
    else:
        print("Test failed: Embedding mismatch detected.")

def main():
    parser = ArgumentParser()
    parser.add_argument('--smiles_path', type=str, required=True)
    parser.add_argument('--smiles_column', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--len_test', type=int, default=0)
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    print("Processing unique SMILES...")
    unique_smiles_list, inverse_mapping, original_length = process_smiles(args.smiles_path, args.smiles_column)

    print(f"Generating embeddings for {len(unique_smiles_list)} items...")
    unique_embeddings_array = generate_embeddings(unique_smiles_list, model, tokenizer, device)

    embeddings = process_embeddings(unique_embeddings_array, inverse_mapping, original_length)

    if args.len_test > 0:
        print("Running consistency test...")
        smiles_df = pd.read_csv(args.smiles_path)
        test(embeddings, smiles_df, args.smiles_column, args.len_test, model, tokenizer, device)
    
    print(f"Saving embeddings to {args.output_path}...")
    np.savez_compressed(args.output_path, embeddings=embeddings)

if __name__ == "__main__":
    main()
