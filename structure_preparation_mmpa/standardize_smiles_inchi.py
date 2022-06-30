from ChEMBL_Structure_Pipeline.chembl_structure_pipeline import standardizer
from rdkit import Chem
from rdkit.Chem import rdDepictor
import pandas as pd
import os
import argparse

def smi_output(data_list):
  smiles = data_list['Smiles'].values.tolist()
  cid = data_list['ChEMBL_ID'].values.tolist()
  n = len(smiles)
  cid_number = []
  inc=[]
  key=[]
  for i in range(n):
    try:
      mol = Chem.MolFromSmiles(smiles[i])
      rdDepictor.Compute2DCoords(mol)
      Chem.WedgeMolBonds(mol, mol.GetConformer())
      molb = Chem.MolToMolBlock(mol)
      std_molblock = standardizer.standardize_molblock(molb)
      inchi=Chem.inchi.MolBlockToInchi(std_molblock)
      inc.append(inchi)
      inchikey=Chem.inchi.InchiToInchiKey(inchi)
      key.append(inchikey)
      cid_number.append(cid[i])
    except:
      pass
  a = {'id':cid_number,'Inchi':inc,'Inchikey':key}
  data_a = pd.DataFrame(a)
  data_a.to_csv(outPath,sep='\t',header=True,index=True)

def sdf_output(sdf_file):
  suppl = Chem.SDMolSupplier(sdf_file)
  inc=[]
  key = []
  id_number = []
  for mol in suppl:
    try:
      rdDepictor.Compute2DCoords(mol)
      Chem.WedgeMolBonds(mol, mol.GetConformer())
      molb = Chem.MolToMolBlock(mol)
      std_molblock = standardizer.standardize_molblock(molb)
      inchi = Chem.inchi.MolBlockToInchi(std_molblock)
      id = mol.GetProp('_Name')
      inchikey = Chem.inchi.InchiToInchiKey(inchi)
      inc.append(inchi)
      key.append(inchikey)
      id_number.append(id)
    except:
	    pass
  a = {'id':id_number,'inchi':inc,'inchikey':key}
  data_a = pd.DataFrame(a)
  data_a.to_csv(outPath, sep='\t', header=True, index=True)


file_path_now=os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("-smi_sdf_Path", help="smi or sdf file located", type=str,default=file_path_now+ "/test_data/assayidchembl976580.csv")
parser.add_argument("-outPath", help="output file located", type=str,default=file_path_now+"/test_data/inchi.tsv")

args = parser.parse_args()
inputPathname1 = args.smi_sdf_Path
outPath = args.outPath

file_type=inputPathname1.split('.')[-1]
if file_type=='csv':
    database_list = pd.read_csv(inputPathname1, header=0, index_col=False, sep=";")
    smi_output(database_list)
elif file_type=='sdf':
    sdf_output(inputPathname1)


