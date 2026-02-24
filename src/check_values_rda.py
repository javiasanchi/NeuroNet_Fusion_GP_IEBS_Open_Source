import pyreadr
import os

RDA_ROOT = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/ida/ADNI/ADNIMERGE2/data'

def check_volumes(filename):
    path = os.path.join(RDA_ROOT, filename)
    result = pyreadr.read_r(path)
    df = result[list(result.keys())[0]]
    cols = ['LHIPPOC', 'RHIPPOC', 'LENTORHIN', 'RENTORHIN', 'LMIDTEMP', 'RMIDTEMP', 'VENTRICLES', 'EICV']
    available_cols = [c for c in cols if c in df.columns]
    print(f"\n--- Checking {filename} ---")
    print(df[available_cols].head())

check_volumes('UCSDVOL.rda')
