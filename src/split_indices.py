from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(path_to_emds: list[Path], output_file: Path) -> None:

    splits = {}

    for strain_path in path_to_emds:
        print(f"Processing train/test indices for {strain_path.name}...")
        
        # Memory-map the array to pull only the target label column
        strain_data = np.load(strain_path, mmap_mode='r')["comb_embs"]
        y = strain_data[:, -1]
        
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        strain_name = str(strain_path.name).split("_")[0]
        splits[strain_name] = {
            "train": train_idx.tolist(), 
            "test": test_idx.tolist()
        }
    
    np.save(output_file, splits)
    print(f"Successfully saved all splits to {output_file}")


def main() -> None:
    parser = ArgumentParser(description="Split strain data into stratified train and test indices.")
    parser.add_argument(
        "--path_to_emds", 
        required=True,
        type=Path,
        help="Directory path containing the .npz strain files."
    )
    parser.add_argument(
        "--output_file", 
        default=Path("data/all_splits.npy"), 
        type=Path,
        help="Output filepath for the train-test splits dictionary."
    )
    args = parser.parse_args()

    # Efficiently gather all matching .npz paths
    npz_files = sorted(args.path_to_emds.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {args.path_to_emds}")
        return

    split_indices(npz_files, output_file=args.output_file)


if __name__ == "__main__":
    main()