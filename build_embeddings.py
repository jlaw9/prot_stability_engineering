import argparse
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
print(f"{torch.cuda.is_available() = }")


# generate the embeddings for these sequences
def get_seq_embeddings(model, seqs, repr_layer=33, batch_size=16):
    """
    Generate an embedding for every sequence using the specified model
    """
    scaler = GradScaler()
    batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(np.arange(len(seqs)), seqs)))

    batch_dataloader = torch.utils.data.DataLoader(batch_tokens, 
                                                   batch_size=batch_size, 
                                                   pin_memory=True,
                                                   num_workers=8
                                                  )

    representations = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(batch_dataloader), total=len(batch_dataloader)):
            out = model(batch.to(device), repr_layers=[repr_layer])  # because this is the 33-layer transformer
            out = out['representations'][repr_layer].detach().to('cpu').numpy()

            for i in range(len(batch)):
                seq_idx = (batch_idx * batch_size) + i
                representations += [out[i, 1:len(seqs[seq_idx]) + 1].mean(0).astype(np.float16)]

            if batch_idx == 0 or batch_idx % int(len(batch_dataloader) / 20.0) == 0:
                # keep track of how much memory this script is using
                print_memory_usage()
    representations = np.vstack(representations)
    return representations


def print_memory_usage():
    # this prints the total memory usage of the machine
    # TODO get the memory usage of this script only
    command = "free -h | head -n 2"
    os.system(command)


aa_checker = re.compile('^[acdefghiklmnpqrstvwy]*$', re.I)

out_dir = "/projects/robustmicrob/jlaw/inputs/meltome"
print(f"reading {out_dir}/20230125_meltome_flip.csv")
df = pd.read_csv(f"{out_dir}/20230125_meltome_flip.csv", index_col=0)
print(len(df))
print(df.head(2))

df = df[df.sequence.apply(lambda seq: aa_checker.search(seq) is not None)]
print(f"removing sequences with non-natural AAs: {len(df)} remaining")

df = df[df.sequence.apply(len) < 1500]
print(f"restricting to sequences with len < 1500: {len(df)} remaining")

#seq_labels = df['uniprot'].values
seqs = df['sequence'].values
# sequence length limit for esm when training
seqs = [seq[:1022] if len(seq) > 1022 else seq for seq in seqs]


torch.hub.set_dir('/scratch/jlaw/torch')
#model_name = "esm2_t33_650M_UR50D"
model_name = "esm2_t36_3B_UR50D"
model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.eval()  # disables dropout for deterministic results
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
# get the representations from the last layer 
repr_layer = model.num_layers

print(f"building embeddings for {len(seqs)} embeddings using {repr_layer = }")
print("current memory usage:")
print_memory_usage()
with autocast():
    representations = get_seq_embeddings(model, seqs, repr_layer=repr_layer, batch_size=1)
print(f"{representations.shape = }")

# write the representations to file
out_file = f"{out_dir}/{embeddings}/20230125_embeddings_{model_name}.npz"
print(f"Writing embeddings to {out_file}")
np.savez(out_file, representations)

df.to_csv(f"{out_dir}/{embeddings}/20230125_embeddings_seqs.csv")

