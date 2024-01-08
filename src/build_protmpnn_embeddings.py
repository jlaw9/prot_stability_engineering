import argparse
import json, time, os, sys, glob, re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange
tqdm.pandas()

# git clone https://github.com/dauparas/ProteinMPNN.git"
# add to path
base_dir = "/projects/bpms/jlaw/tools/ProteinMPNN"
sys.path.append(base_dir)

# Setup Model
import warnings
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, StructureLoader, ProteinMPNN

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#v_48_010=version with 48 edges 0.10A noise
model_name = "v_48_010" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

# Standard deviation of Gaussian noise to add to backbone atoms
backbone_noise=0.0  

path_to_model_weights = f'{base_dir}/vanilla_model_weights'          
hidden_dim = 128
num_layers = 3 
model_folder_path = path_to_model_weights
if model_folder_path[-1] != '/':
    model_folder_path = model_folder_path + '/'
checkpoint_path = model_folder_path + f'{model_name}.pt'

checkpoint = torch.load(checkpoint_path, map_location=device) 
print('Number of edges:', checkpoint['num_edges'])
noise_level_print = checkpoint['noise_level']
print(f'Training noise level: {noise_level_print}A')
model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded")

inputs_dir = Path("/projects/robustmicrob/jlaw/inputs/")

meltome_structures = {}
for pickle_file in glob.glob(f"{inputs_dir}/structures/*.p"):
    structures = pickle.load(open(pickle_file, 'rb'))
    print(f"{len(structures)} read from {pickle_file}")
    if 'meltome_processed.p' in pickle_file:
        structures = {os.path.basename(file_path).split('-')[1]: x 
                      for file_path, x in structures.items()}
    meltome_structures.update(structures)
print(len(meltome_structures))

pdb_dict_list = []
for u_id, dict_list in meltome_structures.items():
    pdb_dict = dict_list[0]
    pdb_dict['uniprot_id'] = u_id
    pdb_dict['name'] = u_id
    pdb_dict_list.append(pdb_dict)

max_length = 5000
# number of tokens for one batch
# batch_size = 10000
# batch_size = 5000
batch_size = 5000

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

chain_id_dict = None
fixed_positions_dict = None
pssm_dict = None
omit_AA_dict = None
bias_AA_dict = None
tied_positions_dict = None
bias_by_res_dict = None
bias_AAs_np = np.zeros(len(alphabet))

def my_featurize(batch, device):
    homomer = False #@param {type:"boolean"}
    designed_chain = "A" #@param {type:"string"}
    fixed_chain = "" #@param {type:"string"}

    designed_chain_list = ["A"]
    fixed_chain_list = []
    chain_list = list(set(designed_chain_list + fixed_chain_list))

    chain_id_dict = {pdb_dict['name']: (designed_chain_list, fixed_chain_list) for pdb_dict in batch}
    tied_positions_dict = None

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
    visible_list_list, masked_list_list, masked_chain_length_list_list, \
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
    tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
    bias_by_res_all, tied_beta = \
        tied_featurize(
    batch, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, \
    tied_positions_dict, pssm_dict, bias_by_res_dict)
    
    return X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all


# create the dataset objects
dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
loader = StructureLoader(dataset, batch_size=batch_size)

# Build the embeddings by taking the last layer before log_probs
#representations = []
max_seq_length = 1500
aa_embed = np.zeros([len(dataset), max_seq_length, 128])
seqs = []
with torch.no_grad():
    idx = 0
    for batch in tqdm(loader):
        start_batch = time.time()
        X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = my_featurize(batch, device)
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs, h_V = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, return_embedding=True)
        h_V = h_V.detach().to('cpu').numpy()
        
        for i in range(len(batch)):
            #representations += [h_V[i, :len(batch[i]['seq'])].mean(0).astype(np.float16)]
            # save the full embeddings
            #representations += [h_V]
            aa_embed[idx, :h_V.shape[1], :] = h_V[i]
            seqs += [(batch[i]['uniprot_id'], batch[i]['seq'])]
            idx += 1

#representations = np.vstack(representations)
representations = aa_embed
print(representations.shape)

# write the representations to file
out_file = f"{inputs_dir}/structures/embeddings/20230221_aa_embeddings_{model_name}.npz"
print(f"Writing embeddings to {out_file}")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, representations)

df = pd.DataFrame(seqs, columns=['uniprot_id', 'sequence'])
df.to_csv(out_file.replace('.npz','.csv'))
#with open(out_file.replace('.npz','.csv'), 'w') as out:
#    out.write('\n'.join([f"{u_id},{seq}" for u_id, seq in seqs]) + '\n')

