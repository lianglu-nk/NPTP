import rdkit
import os 
import sys           
import argparse
from rdkit import Chem
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'mmpa'))
code_path=sys.path[-1]
def process(input_path,out_path1,output2):
    if os.path.exists(out_path1):
        pass
    else:
        command1 = 'python '+code_path+'/rfrag.py <%s >%s' % (input_path,out_path1)
        os.system(command1)
        command2 = 'python '+code_path+'/indexing.py <%s >%s' % (out_path1,output2)
        os.system(command2)

file_path_now=os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("-parent_smi_Path", help="parent smi file located", type=str,default=file_path_now+ "/test_data/parent_smiles.tsv")
parser.add_argument("-fragmentoutPath", help="fragment output file located", type=str,default=file_path_now+"/test_data/sample_fragmented.txt")
parser.add_argument("-indexoutPath", help="index output file located", type=str,default=file_path_now+"/test_data/sample_mmps_default.csv")

args = parser.parse_args()
inputPathname1 = args.parent_smi_Path
fragmentoutPath = args.fragmentoutPath
indexoutPath = args.indexoutPath

process(inputPathname1,fragmentoutPath,indexoutPath)

