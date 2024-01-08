import os
import subprocess
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm 

prots_remaining = pd.read_csv("prots_remaining.txt", squeeze=True)
print(prots_remaining.head(2))
print(len(prots_remaining))

out_dir = "/projects/robustmicrob/jlaw/inputs/structures/meltome"
# for uniprot-ids without a structure, try downloading the alphafold structure
failed = set()
for u_id in tqdm(prots_remaining):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{u_id}-F1-model_v4.pdb"
    out_file = Path(out_dir, f"AF-{u_id}-F1-model_v4.pdb")
    gzipped_file = Path(out_file, '.gz')
    if not gzipped_file.is_file():
        command = f"wget -O {out_file} {url}"
        # print(command)
        try:
            subprocess.check_call(command, shell=True)
            subprocess.check_call(['gzip', out_file])
        except subprocess.CalledProcessError:
            failed.add(u_id)
            os.remove(out_file)
print(f"{len(failed)} structures failed to download")
