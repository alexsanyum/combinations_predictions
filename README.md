# Prediction od additive and non-additive combinations 

## 1. Data download

The models are built based on the dataset described in study of Tse et al. The combination data is available in the Harvard Dataverse. We recommend to manuall download the dataset that contain the Bliss sum scores values into the ```data/``` folder. 

## 2. Quality filter and SMILES normalization 

The script ```process_valid_SMILES_comb.py``` will filter data by quality, and process SMILES in the dataset. The script uses the module ```chem_utilities.py``` to standirize the SMILES according to the next protocolusing functions from ```RDkit```
1. ```TautomerEnumerator’``` and ```’Canonicalize``` to obtain the canonical tautomer
2. ```MetalDisconnector```
3. ```LargestFragmentChooser```
4. ```Uncharger```
5. ```AddHs```

After normalization, the canonical SMILES were written, and combination data is filtered to include only rows with valid SMILES. 

The commnand line to perform this step is the next one:
~~~python
python code/process_valid_SMILES_comb.py --comb_data_path data/B_blisssum_DropArray.csv \
    --abx_data_path metadata/antibiotics_names.txt \
    --output_path data/
~~~

This script will generate three files in the ```data/``` directory
1. ```valid_comb_data.csv``` that contains the filtered small molecule/antibiotics combinations data
2. ```normalized_small_mol.csv``` that contains the normalized SMILES of the small molecules
3. ```normalized_abx.csv``` that contains the normalized SMILES of the antibiotics

## 3. Embeddings calculation

