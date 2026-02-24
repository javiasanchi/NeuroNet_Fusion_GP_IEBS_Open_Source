
import pandas as pd
import pyreadr
import os

RDA_ROOT = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/ida/ADNI/ADNIMERGE2/data'

def check_rda(filename):
    print(f"\n--- Checking {filename} ---")
    path = os.path.join(RDA_ROOT, filename)
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return
    try:
        result = pyreadr.read_r(path)
        # result is a dictionary, usually with one key same as filename base
        for key in result.keys():
            df = result[key]
            print(f"Table: {key}")
            print(f"Columns: {list(df.columns)}")
            # Search for ICV, MidTemp, Ventricles related columns
            relevant = [c for c in df.columns if any(x in c.upper() for x in ['ICV', 'TIV', 'TEMP', 'VENT', 'HIPPO', 'ENTO'])]
            print(f"Potential relevant columns: {relevant}")
            print(f"First row: \n{df.iloc[0] if len(df) > 0 else 'Empty'}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

check_rda('UCSDVOL.rda')
check_rda('UCSFFSX.rda')
check_rda('UCSFFSX7.rda')
