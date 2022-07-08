#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import re
import sys

import pyro
import pyro.distributions as dist

import torch
import torch.nn as nn
import torch.nn.functional as F

global K # Number of topics.
global M # Number of batches.
global N # Number of genes.
global G # Number of genes.

# Set the umber of topics now.
K = 2


SAHA = frozenset([
      "P2449_N724.S508", "P2771_N707.S508", "P2449_N720.S510",
      "P2769_N702.S508", "P2449_N721.S508", "P2771_N710.S507",
      "P2458_N710.S520", "P2770_N701.S518", "P2458_N715.S520",
      "P2458_N714.S517", "P2449_N724.S506", "P2458_N715.S518",
      "P2449_N718.S506", "P2770_N707.S520", "P2770_N704.S520",
      "P2458_N703.S518", "P2770_N711.S520", "P2769_N706.S503",
      "P2458_N702.S521", "P2769_N714.S506", "P2458_N712.S522",
      "P2449_N716.S506", "P2770_N714.S518", "P2769_N714.S507",
      "P2771_N714.S506", "P2770_N704.S518", "P2458_N706.S518",
      "P2771_N704.S503", "P2771_N707.S507", "P2770_N705.S520",
      "P2770_N706.S515", "P2769_N707.S502", "P2770_N711.S513",
      "P2770_N701.S520", "P2458_N712.S518", "P2770_N715.S513",
      "P2458_N711.S522", "P2769_N701.S508", "P2770_N714.S513",
      "P2771_N706.S503", "P2449_N722.S508", "P2458_N710.S518",
      "P2449_N722.S506", "P2769_N706.S507", "P2458_N705.S522",
      "P2458_N703.S520", "P2458_N710.S517", "P2770_N715.S517",
      "P2458_N707.S518", "P2769_N712.S502", "P2458_N715.S522",
      "P2771_N705.S508", "P2458_N710.S522", "P2769_N710.S508",
      "P2458_N711.S517", "P2769_N706.S508", "P2770_N706.S518",
      "P2769_N707.S507", "P2458_N705.S518", "P2449_N727.S506",
      "P2449_N721.S506", "P2771_N701.S503", "P2449_N728.S506",
      "P2449_N729.S506", "P2771_N711.S507", "P2771_N714.S502",
      "P2769_N703.S507", "P2458_N701.S522", "P2458_N714.S521",
      "P2458_N706.S520", "P2449_N722.S510", "P2449_N728.S507",
      "P2449_N719.S508", "P2771_N714.S507", "P2770_N701.S515",
      "P2458_N707.S517", "P2769_N705.S503", "P2769_N711.S502",
      "P2449_N716.S508", "P2770_N710.S517", "P2769_N715.S502",
      "P2449_N722.S507", "P2770_N710.S513", "P2458_N701.S518",
      "P2770_N707.S513", "P2771_N703.S508", "P2458_N711.S521",
      "P2771_N703.S507", "P2770_N702.S520", "P2770_N707.S518",
      "P2771_N702.S507", "P2771_N711.S502", "P2458_N707.S521",
      "P2449_N723.S506", "P2771_N712.S502", "P2449_N726.S506",
      "P2449_N729.S511", "P2769_N711.S507", "P2771_N706.S507",
      "P2770_N714.S517", "P2769_N704.S508", "P2771_N706.S508",
      "P2449_N722.S511", "P2449_N720.S511", "P2770_N712.S517",
      "P2771_N704.S507", "P2458_N710.S521", "P2769_N715.S506",
      "P2770_N703.S520", "P2769_N704.S503", "P2449_N720.S508",
      "P2449_N727.S508", "P2770_N702.S518", "P2770_N704.S515",
      "P2770_N703.S518", "P2771_N710.S508", "P2770_N703.S515",
      "P2449_N720.S507", "P2458_N704.S518", "P2771_N710.S506",
      "P2458_N704.S521", "P2771_N701.S508", "P2449_N719.S506",
      "P2449_N718.S507", "P2771_N715.S506", "P2771_N705.S507",
      "P2449_N719.S507", "P2458_N705.S521", "P2449_N718.S510",
      "P2458_N712.S520", "P2769_N705.S507", "P2458_N703.S517",
      "P2769_N710.S507", "P2449_N716.S510", "P2458_N714.S518",
      "P2449_N724.S507", "P2449_N727.S507", "P2458_N701.S520",
      "P2771_N702.S508", "P2771_N710.S502", "P2769_N703.S508",
      "P2771_N711.S506", "P2449_N716.S507", "P2770_N711.S518",
      "P2449_N726.S510", "P2769_N703.S503", "P2449_N726.S511",
      "P2771_N702.S503", "P2769_N707.S506", "P2458_N707.S522",
      "P2449_N727.S511", "P2449_N723.S507", "P2449_N729.S507",
      "P2769_N710.S502", "P2449_N723.S510", "P2769_N702.S507",
      "P2770_N712.S513", "P2449_N729.S508", "P2771_N707.S502",
      "P2458_N715.S517", "P2449_N718.S511", "P2458_N714.S520",
      "P2458_N707.S520", "P2769_N710.S506", "P2769_N701.S507",
      "P2770_N706.S520", "P2458_N706.S517", "P2771_N701.S507",
      "P2458_N703.S521", "P2771_N711.S508", "P2458_N704.S517",
      "P2769_N715.S507", "P2458_N702.S522", "P2458_N704.S520",
      "P2770_N711.S517", "P2770_N705.S515", "P2458_N711.S520",
      "P2449_N727.S510", "P2449_N728.S508", "P2769_N702.S503",
      "P2458_N706.S521", "P2449_N723.S511", "P2449_N721.S511",
      "P2458_N705.S517", "P2458_N715.S521", "P2771_N712.S507",
      "P2449_N729.S510", "P2770_N707.S517", "P2449_N724.S511",
      "P2458_N702.S517", "P2771_N704.S508", "P2458_N702.S520",
      "P2458_N705.S520", "P2458_N711.S518", "P2769_N712.S507",
      "P2771_N715.S507", "P2770_N710.S518", "P2449_N728.S511",
      "P2458_N701.S517", "P2769_N712.S506", "P2449_N724.S510",
      "P2458_N706.S522", "P2771_N712.S506", "P2458_N702.S518",
      "P2771_N715.S502", "P2458_N704.S522", "P2771_N705.S503",
      "P2458_N701.S521", "P2771_N707.S506", "P2449_N719.S511",
      "P2449_N726.S508", "P2769_N701.S503", "P2458_N712.S521",
      "P2769_N705.S508", "P2769_N711.S508", "P2449_N719.S510",
      "P2449_N721.S507", "P2770_N712.S518", "P2449_N726.S507",
      "P2770_N702.S515", "P2458_N703.S522", "P2770_N710.S520",
      "P2769_N714.S502", "P2449_N716.S511", "P2769_N707.S508",
      "P2770_N715.S518", "P2458_N712.S517", "P2458_N714.S522",
      "P2449_N728.S510", "P2449_N723.S508", "P2769_N704.S507",
      "P2771_N703.S503", "P2449_N720.S506", "P2449_N718.S508",
      "P2449_N721.S510", "P2770_N705.S518", "P2769_N711.S506",
])

def model(data=None):
   # Split data over cell, batch, cell type and expresion.
   cells, batch, ctype, X = data

   # Set global parameter.
   batch_effects = pyro.param("batch_effects",
         torch.ones(M-1, G).to(X.device),
         constraint = torch.distributions.constraints.positive
   )
   type_effects = pyro.param("type_effects",
         torch.ones(N-1, G).to(X.device),
         constraint = torch.distributions.constraints.positive
   )

   # Sample globals.
   with pyro.plate("topics", K):
      # Sample global frequencies of the topics.
      one_over_K = torch.ones(1).to(X.device) / K
      topic_weights = pyro.sample(
            name = "alpha",
            fn = dist.Gamma(one_over_K, 1)
      )
      # Sample word frequencies by topic.
      one_over_G = torch.ones(G).to(X.device) / G
      word_freqs = pyro.sample(
            name = "phi",
            fn = dist.Dirichlet(one_over_G)
      )

   # Sample locals.
   ndocs = X.shape[0]
   with pyro.plate("documents", ndocs):
      # Sample topic frequencies in document.
      doc_topics = pyro.sample(
            name = "theta",
            fn = dist.Dirichlet(topic_weights)
      )
      # Sampling a topic then a word is equivalent
      # to sampling a word from weighted frequencies.
      freqs = torch.mm(doc_topics, word_freqs)
      # Apply batch effects.
      batch_mat = torch.cat([torch.ones(1,G).to(X.device), batch_effects])
      freqs *= torch.mm(F.one_hot(batch).float(), batch_mat)
      # Apply type effects.
      type_mat = torch.cat([torch.ones(1,G).to(X.device), type_effects])
      freqs *= torch.mm(F.one_hot(ctype).float(), type_mat)
      # Sample word counts in document.
      data = pyro.sample(
            name = "g",
            fn = dist.Multinomial(probs = freqs, validate_args = False),
            obs = X
      )

   return topic_weights, word_freqs, data


def guide(data=None):
   # Split data over cell, batch, cell type and expresion.
   cells, batch, ctypes, X = data

   # Use a conjugate guide for global variables.
   topic_weights_posterior = pyro.param(
         "topic_weights_posterior",
         lambda: torch.ones(K, device=X.device),
         constraint = torch.distributions.constraints.positive
   )
   word_freqs_posterior = pyro.param(
         "word_freqs_posterior",
         lambda: torch.ones((K, G), device=X.device),
         constraint = torch.distributions.constraints.greater_than(0.5)
   )

   with pyro.plate("topics", K):
      alpha = pyro.sample(
            name = "alpha",
            fn = dist.Gamma(topic_weights_posterior, 1.)
      )
      phi = pyro.sample(
            name = "phi",
            fn = dist.Dirichlet(word_freqs_posterior)
      )

   ndocs = X.shape[0]
   doc_topics = pyro.param(
         "doc_topic_posterior",
         lambda: torch.ones((ndocs, K), device = X.device),
         constraint = torch.distributions.constraints.greater_than(0.5)
   )
   with pyro.plate("documents", ndocs):
      pyro.sample(
            name = "theta",
            fn = dist.Dirichlet(doc_topics)
   )


def sc_data(fname, device='cpu'):
   """
   Data for single-cell transcriptome, returns a 3-tuple with a tensor
   of batches as integers, a tensor of cell type and a tensor with read
   counts.
   """
   list_of_cells = list()
   list_of_infos = list()
   list_of_exprs = list()
   # Convenience parsing function.
   # Remove rightmost entry (HIV).
   parse = lambda row: (row[0], row[1], [float(x) for x in row[2:-1]])
   with open(fname) as f:
      ignore_header = next(f)
      for line in f:
         cell, info, expr = parse(line.split())
         if cell not in SAHA: continue
         list_of_cells.append(cell)
         list_of_infos.append(info)
         list_of_exprs.append(torch.tensor(expr))
   # Extract batch (plate) from cell ID.
   list_of_plates = [re.sub(r"_.*", "", x) for x in list_of_cells]
   unique_plates = list(set(list_of_plates))
   list_of_batch_ids = [unique_plates.index(x) for x in list_of_plates]
   # Extract cell type from treatment info.
   list_of_ctypes = [re.sub(r"\+.*", "", x) for x in list_of_infos]
   unique_ctypes = sorted(list(set(list_of_ctypes)))
   list_of_ctype_ids = [unique_ctypes.index(x) for x in list_of_ctypes]
   # Return the (cells, batches, types, expressions) tuple.§
   batch_tensor = torch.tensor(list_of_batch_ids).to(device)
   ctype_tensor = torch.tensor(list_of_ctype_ids).to(device)
   expr_tensor = torch.stack(list_of_exprs).to(device)
   return list_of_cells, batch_tensor, ctype_tensor, expr_tensor


data = sc_data('alivecells.tsv')

pyro.clear_param_store()
torch.manual_seed(123)
pyro.set_rng_seed(123)

# Set global dimensions.
cells, batches, ctypes, X = data
M = batches.max() + 1
N = ctypes.max() + 1
G = X.shape[-1]

optimizer = torch.optim.AdamW
scheduler = pyro.optim.PyroLRScheduler(
      scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR,
      optim_args = {'optimizer': optimizer, 'optim_args': {'lr': 0.01}, 'gamma': 0.1},
      clip_args = {'clip_norm': 5.}
)
svi = pyro.infer.SVI(model, guide, scheduler,
      loss=pyro.infer.Trace_ELBO())


loss = 0.
for step in range(7000):
   loss += svi.step(data)
   if (step+1) % 1000 == 0:
      sys.stderr.write("iter {}: loss = {:.2f}\n".format(step+1, loss / 1e9))
      loss = 0.
   if (step+1) % 6000 == 0:
      scheduler.step()


###
out = pyro.param("doc_topic_posterior")
wfreq = pyro.param("word_freqs_posterior")
batch_effects = pyro.param("batch_effects")
type_effects = pyro.param("type_effects")
# Output signature breakdown with row names.
pd.DataFrame(out.detach().cpu().numpy(), index=cells) \
   .to_csv("out-SAHA-no-HIV.txt", sep="\t", header=False, float_format="%.5f")
np.savetxt("wfreq-SAHA-no-HIV.txt", wfreq.detach().cpu().numpy(), fmt="%.5f")
np.savetxt("batch-effects-SAHA-no-HIV.txt",
      batch_effects.detach().cpu().numpy(), fmt="%.5f")
np.savetxt("type-effects-SAHA-no-HIV.txt",
      type_effects.detach().cpu().numpy(), fmt="%.5f")
