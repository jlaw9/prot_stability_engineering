import gzip
import tarfile
import pickle
import sys
import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = "/projects/bpms/jlaw/tools/ProteinMPNN"
sys.path.append(base_dir)
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN


def read_embeddings(embed_file, sequence_idx_file):
    """ Read embeddings stored in an npz file
    Get the sequences at each index from the *sequence_idx_file
    """
    embeddings = np.load(embed_file, allow_pickle=True)['arr_0']
    sequences = pd.read_csv(sequence_idx_file)
    print(f"{len(embeddings) = } read from {embed_file}")
    print(f"{len(sequences) = } read from {sequence_idx_file}")
    return embeddings, sequences


inputs_dir = "/projects/robustmicrob/jlaw/inputs/"

data_file = "/projects/robustmicrob/jlaw/inputs/meltome/flip/github/full_dataset_sequences.csv.gz"
data = pd.read_csv(data_file)
print(len(data))
print(data.head(2))

# try using the same train/test splits that flip used
df_split = pd.read_csv(Path(inputs_dir, "meltome/flip/github/splits/mixed_split.csv"))
print(len(df_split))
print(df_split.head(2))
seq_to_uniprot = dict(zip(data.sequence, data.uniprot))
df_split['uniprot_id'] = df_split.sequence.apply(lambda seq: seq_to_uniprot[seq])

embeddings, df_seq = read_embeddings(Path(inputs_dir, "meltome/embeddings/20230206_embeddings_esm2_t33_650M_UR50D.npz"),
                                     Path(inputs_dir, "meltome/embeddings/20230125_embeddings_seqs.csv"))
df_split_w_embed = df_split[df_split.sequence.isin(df_seq.sequence)]
print(len(df_split_w_embed))

prot_ids = df_split_w_embed.uniprot_id.unique()
print(f"{len(prot_ids) = }")

prot_with_structure = set()
for strc_tar_file in glob.glob(f"{inputs_dir}/structures/*.tar"):
    prot_structure = {}
    prot_with_structure = set()
    print(strc_tar_file)
    with tarfile.open(strc_tar_file, 'r') as tar:
        prot_structure = {}
        for member in tqdm(tar.getmembers()):
            if 'cif' in member.name:
                continue
            u_id = member.name.split('-')[1]
            if u_id in prot_ids:
                prot_with_structure.add(u_id)
                pdb_file = tar.extractfile(member)
                file_ = gzip.decompress(pdb_file.read()).decode()
                pdb = parse_PDB(file_handle=file_.split('\n'))
                prot_structure[u_id] = pdb

    print(len(prot_with_structure))
    with open(strc_tar_file.replace('.tar','.p'), 'wb') as out:
        pickle.dump(prot_structure, out)


prot_structure = {}
for pdb_file in tqdm(glob.glob(f"{inputs_dir}/structures/meltome/*.pdb.gz")):
    pdb = parse_PDB(path_to_pdb=pdb_file)
    prot_structure[pdb_file] = pdb
print(len(prot_structure))
out_file = f"{inputs_dir}/structures/meltome_processed.p"
with open(out_file, 'wb') as out:
    pickle.dump(prot_structure, out)

print("Finished")
