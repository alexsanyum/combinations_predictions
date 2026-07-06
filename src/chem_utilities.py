# RDkit library for cheminformatics
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit import RDLogger
from rdkit.Chem import AddHs, MolToSmiles
import pandas as pd
from dimorphite_dl import protonate_smiles


def normalize_smiles(smiles, protonate=True):
    """Normalize a SMILES"""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    
    # Discard invalid SMILES
    if "and" in smiles or " + " in smiles:
        return None

    smiles = smiles.strip()
    smiles = smiles.split(" |")[0]  #Extract SMILES before " |"

    # Handle specific SMILES with "|" instead of "."
    if "[Co+3]|[C-]#N" in smiles:
        smiles = smiles.split('.')[1]
    
    try: 
        mol = Chem.MolFromSmiles(smiles)    # pylint: disable=no-member
        if mol is None:
            return None
        #Standardization
        te = TautomerEnumerator()
        mol = te.Canonicalize(mol)          # Canonical Tautomer
        md = rdMolStandardize.MetalDisconnector()
        mol = md.Disconnect(mol)            # Disconect metal atoms
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)               # Largest fragment choose
        # Protonation by pH
        if protonate:
            mol_string = protonate_smiles(MolToSmiles(mol), 
                                          ph_min=7.2, ph_max=7.4, precision=1, max_variants=1)
            mol = Chem.MolFromSmiles(mol_string[0])
        mol = AddHs(mol)               # Explicit Hydrogens
        canon_smile = MolToSmiles(mol, canonical=True)  
        
        logger.setLevel(RDLogger.WARNING)
        return canon_smile 
    
    except Exception as e:
        logger.setLevel(RDLogger.WARNING)
        return None

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def calculate_descriptors(smiles):
    """Calculate molecular descriptors for a SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {}
    for name, func in Descriptors._descList:
        try:
            descriptors[name] = func(mol)
        except:
            descriptors[name] = None
    return descriptors

def calculate_descriptors_df(smiles_df, smiles_column):
    """Calculate molecular descriptors for a DataFrame with SMILES"""
    descriptor_list = []
    for smiles in smiles_df[smiles_column]:
        descriptors = calculate_descriptors(smiles)
        descriptor_list.append(descriptors)
    descriptor_df = pd.DataFrame(descriptor_list)
    result_df = pd.concat([smiles_df.reset_index(drop=True), descriptor_df], axis=1)
    return result_df

def normalize_smiles_df(smiles_df, smiles_column, protonate=True):
    """Normalize a DataFrame with SMILES"""
    smiles_df['normalized_smiles'] = smiles_df[smiles_column].apply(normalize_smiles, protonate=protonate)    
    return smiles_df
