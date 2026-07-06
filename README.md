# Prediction od additive and non-additive combinations 

## 1. Data download

The models are built based on the dataset described in study of Tse et al. The combination data is available in the Harvard Dataverse. We recommend to manuall download the dataset that contain the Bliss sum scores values into the ```data/``` folder. 

## 2. Data pre-processing 

For avoiding re-calculating same CLS embedding from the same SMILES, the next process first filters valids combinations and normalize the SMILES following a set of rules. Then, a temporal list of unique SMILES are generated from the dataset, and CLS embeddings are processed for these. After this, the generated embeddings are concatenated for each combination data. 


## 2.1. Quality filter and SMILES normalization 

The script ```process_valid_SMILES_comb.py``` will filter data by quality, and process SMILES in the dataset. The script uses the module ```chem_utilities.py``` to standirize the SMILES according to the next protocolusing functions from ```RDkit```
1. ```TautomerEnumerator’``` and ```’Canonicalize``` to obtain the canonical tautomer
2. ```MetalDisconnector```
3. ```LargestFragmentChooser```
4. ```Uncharger```
5. ```AddHs```

After normalization, the canonical SMILES were written, and combination data is filtered to include only rows with valid SMILES. 

The commnand line to perform this step is the next one:
~~~bash
python code/process_valid_SMILES_comb.py --comb_data_path data/B_blisssum_DropArray.csv \ 
                                         --abx_data_path metadata/antibiotics_names.txt \
                                         --output_path data/
~~~

This script will generate three files in the ```data/``` directory
1. ```valid_comb_data.csv``` that contains the filtered small molecule/antibiotics combinations data
2. ```normalized_small_mol.csv``` that contains the normalized SMILES of the small molecules
3. ```normalized_abx.csv``` that contains the normalized SMILES of the antibiotics

### 2.2. Embeddings calculation

The script ```smiles2clsembs.py``` load the pre-trained model ```DeepChem/ChemBERTa-77M-MLM``` and process each normalized SMILES. The processing will be performed on GPU if available, and savec on disk.

~~~bash
python code/smiles2clsembs.py --smiles_path data/normalized_abx.csv \
--smiles_column normalized_smiles \
--model_name DeepChem/ChemBERTa-77M-MLM \ 
--output_path data/abx_embs.npz

python code/smiles2clsembs.py --smiles_path data/normalized_small_mol.csv \ 
--smiles_column normalized_smiles \
--model_name DeepChem/ChemBERTa-77M-MLM \
--output_path data/small_mol_embs.npz 
~~~

The script returns a compressed numpy file for antibiotics and small molecules embeddings; 

### 2.3. Concatenating embeddings for strain 
The  ```process_valid_SMILES_comb.py``` script takes previous results and build the concatenation: [antiobitic embedgins | small molecule embedding | bliss score] for each strain, and save into disk. 

~~~bash
python code/concat_strain_embs.py --comb_data data/valid_comb_data.csv \
                                  --norm_small_mols data/normalized_small_mol.csv \
                                  --small_mol_embs data/small_mol_embs.npz \
                                  --norm_abx data/normalized_abx.csv \
                                  --abx_embs data/abx_embs.npz \
                                  --output_dir data/strains_embs/
~~~

Note: The scripts take a couple of minutes, as saves the concatenated as compressed numpy arrays.