
NOTE:Open source scripts for ChEMBL standardization need to be built into the floder 
git clone https://github.com/chembl/ChEMBL_Structure_Pipeline.git
pip install ./ChEMBL_Structure_Pipeline
version:rdkit >= 2020.09.5
1)
##standardize smiles to inchi##
python ./standardize_smiles_inchi.py -smi_sdf_Path <> -outPath <>

This program has several options (see help from program below):

Usage: standardize_smiles_inchi.py [options]

Options:
  -h, --help            show this help message and exit.
  -smi_sdf_Path         smi or sdf file located.
  -outPath              output file located(inchi).

Example commands (with the compounds under assay_id=chembl976580):
python ./standardize_smiles_inchi.py#use default parameters, no need to set#
2)
##standardize smiles to parent smiles##
python ./parent_smiles.py -smi_sdf_Path <> -outPath <>

This program has several options (see help from program below):

Usage: parent_smiles.py [options]

Options:
  -h, --help            show this help message and exit.
  -smi_sdf_Path         smi or sdf file located.
  -outPath              output file located(parent smiles).

Example commands (with the compounds under assay_id=chembl976580):
python ./parent_smiles.py#use default parameters, no need to set#
3)
##use parent smiles to do mmpa##
python ./mmp.py -parent_smi_Path <> -fragmentoutPath <> -indexoutPath <>

This program has several options (see help from program below):

Usage: mmp.py [options]

Options:
  -h, --help            show this help message and exit.
  -parent_smi_Path      parent smi file located.
  -fragmentoutPath      fragment output file located.
  -indexoutPath         index output file located

Example commands (with the compounds under assay_id=chembl976580):
python ./mmp.py#use default parameters, no need to set#

