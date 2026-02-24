import pyreadr
import os

RDA_ROOT = 'E:/MACHINE LEARNING/proyecto_global_IEBS/data/ida/ADNI/ADNIMERGE2/data'

def check_rda(filename):
    path = os.path.join(RDA_ROOT, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"\n--- Checking {filename} ---")
    result = pyreadr.read_r(path)
    df_name = list(result.keys())[0]
    df = result[df_name]
    
    # Search for ICV related columns
    icv_cols = [col for col in df.columns if 'ICV' in col.upper() or 'EICV' in col.upper()]
    print(f"ICV/EICV columns: {icv_cols}")
    
    potential_tiv = [col for col in df.columns if 'TIV' in col.upper()]
    print(f"TIV columns: {potential_tiv}")
    
    mt_cols = [col for col in df.columns if 'MIDTEMP' in col.upper()]
    print(f"MidTemp columns: {mt_cols}")

    vent_cols = [col for col in df.columns if 'VENT' in col.upper()]
    print(f"Ventricle columns: {vent_cols}")

    if icv_cols or potential_tiv or mt_cols or vent_cols:
        print("Sample values for these columns:")
        print(df[icv_cols + potential_tiv + mt_cols + vent_cols].head())

check_rda('UCSDVOL.rda')
check_rda('UCSFFSX.rda')
