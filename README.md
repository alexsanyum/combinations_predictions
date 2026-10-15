# Prediction of additive and non-additive combinations

## 1. Data download

The models are built on the dataset described in the study by Tse et al. The combination data is available in the Harvard Dataverse. We download the dataset containing the Bliss sum scores into the `data/` folder directly from the repository.

## 2. Data pre-processing

To avoid re-calculating embeddings for the same SMILES, the next process first filters valid combinations and normalizes the SMILES following a set of rules. Then, a temporal list of unique SMILES is generated from the dataset, and CLS embeddings are processed for these. After this, the generated embeddings are concatenated for each combination data.

### 2.1. Quality filter and SMILES normalization

The script `process_valid_SMILES_comb.py` will filter data by quality and process SMILES in the dataset. The script uses the module `chem_utilities.py` to standardize the SMILES according to the following protocol using functions from `RDkit`:
1. `TautomerEnumerator` and `Canonicalize` to obtain the canonical tautomer
2. `MetalDisconnector`
3. `LargestFragmentChooser`
4. `Uncharger`
5. `AddHs`

After normalization, the canonical SMILES were written, and combination data is filtered to include only rows with valid SMILES.

The command line to perform this step is the following:
~~~bash
python src/process_valid_SMILES_comb.py --comb_data_path data/B_blisssum_DropArray.csv \ 
                                         --abx_data_path metadata/antibiotics_names.txt \
                                         --output_path data/
~~~

This script will generate three files in the `data/` directory:
1. `valid_comb_data.csv` that contains the filtered small molecule/antibiotics combinations data
2. `normalized_small_mol.csv` that contains the normalized SMILES of the small molecules
3. `normalized_abx.csv` that contains the normalized SMILES of the antibiotics

### 2.2. Embeddings calculation

The script `smiles2clsembs.py` loads the pre-trained model `DeepChem/ChemBERTa-77M-MLM` and processes each normalized SMILES. The processing will be performed on GPU if available, and saved on disk.

~~~bash
python src/smiles2clsembs.py --smiles_path data/normalized_abx.csv \
                             --smiles_column normalized_smiles \
                             --model_name DeepChem/ChemBERTa-77M-MLM \ 
                             --output_path data/abx_embs.npz

python src/smiles2clsembs.py --smiles_path data/normalized_small_mol.csv \ 
                             --smiles_column normalized_smiles \
                             --model_name DeepChem/ChemBERTa-77M-MLM \
                             --output_path data/small_mol_embs.npz 
~~~

The script returns a compressed numpy file for antibiotics and small molecules embeddings.

### 2.3. Concatenating embeddings for strain

The `process_valid_SMILES_comb.py` script takes `small_mol_embs.npz` and `abx_embs.npz` and builds the concatenation: [antibiotic embedding | small molecule embedding | bliss score] for each strain, and saves it to disk.

~~~bash
python src/concat_strain_embs.py --comb_data data/valid_comb_data.csv \
                                 --norm_small_mols data/normalized_small_mol.csv \
                                 --small_mol_embs data/small_mol_embs.npz \
                                 --norm_abx data/normalized_abx.csv \
                                 --abx_embs data/abx_embs.npz \
                                 --output_dir data/strains_embs/
~~~
Note: The scripts take a couple of minutes, as they save the concatenated data as compressed numpy arrays.

The script will save in disk the concatenated embeddings for each strain dataset (six in total) into a `{strain}_embeddings.npz` in the `data/strains_embs` directory. It also saves a csv file per strain containing the antibiotics and small molecule name, and Bliss score.

## 3. Machine Learning model training and evaluation

Five classifiers are evaluated: Logistic Regression, Support Vector Classifier, Random Forest, XGBoost, and LightGBM. Each classifier was evaluated over the six datasets.
For each model, we followed the next steps:
- 1. Split data into train and test indices
- 2. Hyperparameter tuning
- 3. Bootstrap on best model evaluation

### 3.1. Split train and test splits indices

The `split_indices.py` receives the path where the concatenated embeddings for each dataset are located. On each one, it splits into 80% training and 20% testing, storing just the index.

~~~bash 
python src/split_indices.py --path_to_emds data/strains_embs/ \
                            --output_file data/strains_embs/train_test_splits_indices.npy
~~~

The program creates a npy file containing the train and test indices for each dataset.

### 3.2. Hyperparameter tuning

The script `tuning_ml_models.py` evaluates different hyperparameter spaces for each classifier across the six concatenated embeddings files. It integrates the custom cluster-based undersampling method implemented as a python class (`cluster_undersampling.py`). For parameter grid search, it uses the function from the `skopt` library. As target score, we set the IMCP score.

~~~bash
python -u -m src.tuning_ml_models \ 
        --strain_embs "data/strains_embs/*.npz" \
        --splits_path "data/strains_embs/train_test_splits_indices.npy" \
        --output_dir "data/models_production/" \
        --n_iter 30 \
        --cv 5 \
        --model {lr,svc,rf,xgb,lgbm} \
        --n_jobs 3
~~~

We added the --n_jobs parameter that directly affects the multithread behavior of `BayesSearchCV`. The program runs for a specific ML model from the following ones {lr,svc,rf,xgb,lgbm}, over the six strain datasets. The program returns a `.pkl` file per model and strain, that contains the BayesSearchCV object.

### 3.3. Bootstrap iterations

The script `bootstrap_analysis.py` calculates metrics of one model with the best set of hyperparameters (configuration extracted from the pkl file from the previous step), resamples the train dataset, retrains the model, and evaluates different metrics. The process is repeated `N_BOOTSTRAPS` times.

~~~bash
python -u -m src.bootstrap_analysis     --PATH_TO_MODELS "data/models/"     \
                                        --PATH_TO_DATA "data/strains_embs/" \
                                        --PATH_TO_INDICES "data/strains_embs/train_test_splits_indices.npy"   \
                                        --OUTPUT_DIR "data/bootstrap_results_test/"     
                                        --STRAIN "Ab17978"     
                                        --MODEL_NAME "lgbm"     
                                        --N_BOOTSTRAPS 1000
~~~

This step was run just for the lgbm models, but supports other classifiers. At the end, the program returns a npz file containing the metrics obtained on each metric, as well as feature importance values.

### 3.4. Final model training

One last model using the whole training split dataset was used to train the lgbm model with the best hyperparameter configuration. This was done specifically for the lgbm models.

~~~bash
python -m src.retrain_models   --models_dir "data/models_production/"  \
                               --embs_dir "data/strains_embs/"  \
                               --splits_path "data/strains_embs/train_test_splits_indices.npy"   \
                               --output_dir "data/final_models/"
~~~

Final models are saved as pkl files.

### 4. Prediction of unscreened combinations

The dataset contains theoretical antibiotics and small-molecule pairs that were not experimentally evaluated. Those are first identified and constructed (concatenation of embeddings) and used as input for the models.

This is done in the following `predict_unknown_combinations.py` script.

~~~bash
python -m src.predict_unknown_combinations  --abx_norm data/normalized_abx.csv \
                                            --sm_norm data/normalized_small_mol.csv \
                                            --abx_embs data/abx_embs.npz  \
                                            --sm_embs data/small_mol_embs.npz \
                                            --strain_comb data/strains_embs/ \
                                            --models_path data/final_models/ \
                                            --output_dir data/unknown_combs/
~~~

## Results analysis

In the `notebook` directory are Jupyter notebooks in which the results and metrics of the mentioned process are evaluated.

- `tse_data_exploration.ipynb` Analyzes the dataset, and the molecules represented through the chemBERTa embeddings
- `hyperparameter_analysis.ipynb` Analyzes the metrics during hyperparameter tuning for each of the five classifiers
- `precsion_recall_analysis.ipynb` Computes and calculates the precision-recall curve for the six strain models
- `undersampling_impact.ipynb` Analyzes the effect of the custom undersampling method on the dataset diversity and the model performance
- `unknown_comb_predictions.ipynb` Analyzes the results for the unscreened combinations, counting each of the classes.

### Environment configuration 
In the `singulary_configurations` directory is available the dockerfiles for 3 different set ups. Dockerfile.data was used for data preprocessing, SMILES standardization, and results analysis. Dockerfile.embs is specifically for chemBERTa processing. And Docker.ml sets up the environment for machine learning model training.