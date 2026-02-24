import os
import pandas as pd

def find_markers(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path, nrows=1)
                    cols = [c.lower() for c in df.columns]
                    found = [c for c in df.columns if any(m in c.lower() for m in ['abeta', 'tau', 'csf', 'plasma'])]
                    if found:
                        print(f"File: {path}")
                        print(f"  Columns: {found}")
                except:
                    pass

if __name__ == "__main__":
    find_markers('data/ida/ADNI')
    find_markers('data/ida/OASIS3_data_files')
