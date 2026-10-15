# Prediction od additive and non-additive combinations 

## 1. Data download

The models are built on the dataset described in study of Tse et al. The combination data is available in the Harvard Dataverse. We download the dataset that contain the Bliss sum scores values into the ```data/``` folder directly from the repository. 

## 2. Data pre-processing 

For avoiding re-calculating embeddings for the same SMILES, the next process first filters valids combinations and normalize the SMILES following a set of rules. Then, a temporal list of unique SMILES are generated from the dataset, and CLS embeddings are processed for these. After this, the generated embeddings are concatenated for each combination data. 


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
python src/process_valid_SMILES_comb.py --comb_data_path data/B_blisssum_DropArray.csv \ 
                                         --abx_data_path metadata/antibiotics_names.txt \
                                         --output_path data/
~~~

This script will generate three files in the ```data/``` directory
1. ```valid_comb_data.csv``` that contains the filtered small molecule/antibiotics combinations data
2. ```normalized_small_mol.csv``` that contains the normalized SMILES of the small molecules
3. ```normalized_abx.csv``` that contains the normalized SMILES of the antibiotics

### 2.2. Embeddings calculation

The script ```smiles2clsembs.py``` load the pre-trained model ```DeepChem/ChemBERTa-77M-MLM``` and process each normalized SMILES. The processing will be performed on GPU if available, and saved on disk.

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

The script returns a compressed numpy file for antibiotics and small molecules embeddingss 

### 2.3. Concatenating embeddings for strain 

The  ```process_valid_SMILES_comb.py``` script takes ```small_mol_embs.npz``` and ```abx_embs.npz``` and build the concatenation: [antiobitic embedgins | small molecule embedding | bliss score] for each strain, and save into disk. 

~~~bash
python src/concat_strain_embs.py --comb_data data/valid_comb_data.csv \
                                 --norm_small_mols data/normalized_small_mol.csv \
                                 --small_mol_embs data/small_mol_embs.npz \
                                 --norm_abx data/normalized_abx.csv \
                                 --abx_embs data/abx_embs.npz \
                                 --output_dir data/strains_embs/
~~~
Note: The scripts take a couple of minutes, as saves the concatenated as compressed numpy arrays.

The script will save in disk the concatenated embeddings for each strain data set (six in total) into a ```{strain}_embeddings.npz``` in the ```data/strains_embs``` directory. It also saves a csv file per strain containing the antibitoics and small molecule name, and Bliss score.

## 3 Machine Learnig model training and evuation

Five classifiers are evaluted: Logistic Regression, Support Vector Classiier, Random Forest, XGBoost, and LigthGBM. Each classifier was evaluated over the six dataset. 
For each model, we folloed next steps
- 1. Split data intro train and test indices
- 2. Hyperaprametner tuning 
- 3. Bootstrap on best model evaluation
  
### 3.1. Split train and test splits indices

The ```split_indices.py``` recieves the path were the concatenated embeddings for each dataset are located. On each one, it split into 80% training and 20% testing storing just the index. 

~~~bash 
python src/split_indices.py --path_to_emds data/strains_embs/ \
                            --output_file data/strains_embs/train_test_splits_indices.npy
~~~

The program creates a npy file contaiint the train and test indeces for each dataset. 

### 3.2 Hyperparameter tuning

The script ```tuning_ml_models.py``` evaluate different hyperparamaters space for each classifiers across the six concatenated embeddings files. It integrates the custom cluster-based undersampling method implemented as a python class (```cluster_undersampling.py```). For parameter grid serach, it is uses the  fucntion from the ```skopt``` library. As target score, we set the IMCP score.

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

We added the --n_jobs parameter that directly afects the multithread behabiour of ```BayesSearchCV```. The program runs for a specif ML model from the next ones {lr,svc,rf,xgb,lgbm}, over the six strain datasets. The program returns a ```.pkl``` file per model and strain, that contains the BayesSearchCV object 

### 3.3. Bootstrap iterations

The scripts ```bootstrap_analysis.py``` calculates metrics of one model with the best set of hyperparamters (condiguration extracted from the pkl file from previous step), resample the train dataset, retrain the model, and evaluate different metrics. Process is repeated  ```N_BOOTSTRAPS``` times


~~~bash
python -u -m src.bootstrap_analysis     --PATH_TO_MODELS "data/models/"     \
                                        --PATH_TO_DATA "data/strains_embs/" \
                                        --PATH_TO_INDICES "data/strains_embs/train_test_splits_indices.npy"   \
                                        --OUTPUT_DIR "data/bootstrap_results_test/"     
                                        --STRAIN "Ab17978"     
                                        --MODEL_NAME "lgbm"     
                                        --N_BOOTSTRAPS 1000
~~~

This step was run just for the lgbm models, but supports other classifiers. At the end, the program returns a npz file containing the metrics obtained on each metrics, as well as feature importance value.

### 3.4. Final model training

One last model using the whole training split dataset was used to train lgbm model with the best hyperparameter condiguration. This was done specifically for the lgbm models.

~~~bash
python -m src.retrain_models   --models_dir "data/models_production/"  \
                               --embs_dir "data/strains_embs/"  \
                               --splits_path "data/strains_embs/train_test_splits_indices.npy"   \
                               --output_dir "data/final_models/"
~~~

Final models are saved as pkl files.

### 4. Prediction of unscreened combinations

The dataset contains theorical antibiotics and small-molecules pairs that were not experimentally evaluated. Those are first identified and constructed (concatenation of embeddings) and uses as input for the models. 
 
This is done in the next ```predict_unknown_combinations.py``` script.

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

In the ```notebook``` directory are jupyter-notebook in which are evaluated the results and metrics of  the mentioned process.

- ```tse_data_exploration.ipynb``` Analyze the dataset, and the molecules represented through the chemBERTa embeddings
- ```hyperparameter_analysis.ipynb``` Analyze the metrics during hyperparamter tunning for each of the five classifiers
- ```precsion_recall_analysis.ipynb``` Computes and calculate the precion-recall curve for the six strains models
- ```undersampling_impact.ipynb``` analyze the effect of the custom undersampling method on the dataset diversity and the model performance
- ```unknown_comb_predictions.ipynb``` Anylyze the results for the unscreened combinations, counting each of the classes. 

### Envioroment configuration 
In the ```singulary_configurations``` direcory is available the dockerfiles for 3 different set ip. Dockerfile.data was used for data preprocessing, SMILES standardization, and results analysis. Dockerfile.embs is specifcally for chemBERTa processing. And Docker.ml setup envrioment for machine learning model training. 