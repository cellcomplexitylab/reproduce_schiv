#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import random
import re
import sys

import pyro
import pyro.distributions as dist

import torch
import torch.nn as nn
import torch.nn.functional as F

global K # Number of topics.
global M # Number of batches.
global N # Number of types.
global G # Number of genes.

# Set the umber of topics now.
K = 2

# Those are all Jurkat and J-LatA2 cells treated with PMA,
# including those that are dead. But the input data is only
# the cells that are alive. The list below is used only for
# filtering.
PMA = frozenset([
      "P2769_N710.S503", "P2769_N711.S510", "P2771_N706.S511",
      "P2770_N711.S522", "P2770_N705.S521", "P2771_N702.S510",
      "P2769_N701.S505", "P2770_N703.S516", "P2771_N714.S510",
      "P2769_N701.S510", "P2769_N706.S511", "P2769_N711.S503",
      "P2769_N705.S511", "P2771_N715.S503", "P2771_N715.S511",
      "P2771_N707.S503", "P2769_N705.S510", "P2771_N714.S508",
      "P2770_N707.S521", "P2771_N701.S511", "P2770_N710.S522",
      "P2769_N714.S508", "P2769_N715.S510", "P2769_N712.S510",
      "P2770_N704.S516", "P2769_N704.S505", "P2769_N712.S508",
      "P2769_N714.S510", "P2770_N703.S521", "P2771_N712.S503",
      "P2770_N705.S522", "P2770_N701.S522", "P2769_N704.S510",
      "P2770_N714.S521", "P2770_N703.S522", "P2771_N710.S511",
      "P2771_N712.S511", "P2770_N707.S515", "P2770_N710.S521",
      "P2770_N711.S515", "P2771_N702.S511", "P2769_N702.S511",
      "P2771_N704.S505", "P2771_N711.S503", "P2771_N703.S505",
      "P2769_N707.S503", "P2771_N707.S511", "P2771_N704.S510",
      "P2770_N715.S521", "P2770_N714.S520", "P2771_N704.S511",
      "P2771_N711.S511", "P2771_N703.S511", "P2769_N707.S511",
      "P2769_N703.S510", "P2771_N715.S508", "P2769_N701.S511",
      "P2769_N715.S503", "P2770_N702.S522", "P2770_N706.S522",
      "P2770_N714.S522", "P2769_N710.S510", "P2771_N705.S511",
      "P2771_N702.S505", "P2769_N705.S505", "P2771_N705.S510",
      "P2769_N703.S511", "P2770_N712.S522", "P2771_N712.S508",
      "P2770_N701.S521", "P2769_N710.S511", "P2771_N714.S503",
      "P2769_N704.S511", "P2770_N706.S516", "P2770_N712.S521",
      "P2771_N715.S510", "P2769_N702.S505", "P2770_N715.S522",
      "P2771_N703.S510", "P2770_N705.S516", "P2771_N714.S511",
      "P2770_N702.S521", "P2770_N712.S515", "P2769_N712.S511",
      "P2769_N706.S510", "P2771_N701.S510", "P2770_N711.S521",
      "P2769_N711.S511", "P2769_N706.S505", "P2771_N710.S503",
      "P2770_N707.S522", "P2770_N701.S516", "P2770_N710.S515",
      "P2771_N711.S510", "P2771_N706.S510", "P2770_N715.S515",
      "P2769_N712.S503", "P2771_N707.S510", "P2769_N715.S508",
      "P2770_N704.S522", "P2771_N712.S510", "P2769_N714.S503",
      "P2769_N702.S510", "P2770_N714.S515", "P2771_N701.S505",
      "P2769_N707.S510", "P2771_N710.S510", "P2770_N706.S521",
      "P2770_N712.S520", "P2769_N715.S511", "P2770_N715.S520",
      "P2771_N706.S505", "P2770_N704.S521", "P2769_N714.S511",
      "P2770_N702.S516", "P2769_N703.S505", "P2771_N705.S505",
])

def model(data=None):
   # Split data over cell, batch, cell type and expresion.
   cell, batch, ctype, X = data

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
   cell, batch, ctypes, X = data

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
   parse = lambda row: (row[0], row[1], [float(x) for x in row[2:]])
   with open(fname) as f:
      ignore_header = next(f)
      for line in f:
         cell, info, expr = parse(line.split())
         if cell not in PMA: continue
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
   # Return the (cells, batches, types, expressions) tuple.
   batch_tensor = torch.tensor(list_of_batch_ids).to(device)
   ctype_tensor = torch.tensor(list_of_ctype_ids).to(device)
   expr_tensor = torch.stack(list_of_exprs).to(device)
   return list_of_cells, batch_tensor, ctype_tensor, expr_tensor


data = sc_data('alivecells.tsv')

# Set global dimensions.
cells, batches, ctypes, X = data
M = batches.max() + 1
N = ctypes.max() + 1
G = X.shape[-1]

# Run 210 repetitions where HIV is shuffled.
# The first 10 are not shuffled.
# The next 100 are shuffled by plate (but within cell type).
# The next 100 are shuffled globally (still within cell type).
for iterno in range(210):
   if iterno > 109:
      # Shuffle HIV globally (updates X in place).
      for ctype in range(N):
         idx, = torch.where(ctypes == ctype)
         X[idx,-1] = X[idx[torch.randperm(len(idx))],-1]
   elif iterno > 9:
      # Shuffle HIV by plate (updates X in place).
      for batch in range(M):
         for ctype in range(N):
            idx, = torch.where((batches == batch) & (ctypes == ctype))
            X[idx,-1] = X[idx[torch.randperm(len(idx))],-1]
   pyro.clear_param_store()
   optimizer = torch.optim.AdamW
   scheduler = pyro.optim.PyroLRScheduler(
         scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR,
         optim_args = {'optimizer': optimizer, 'optim_args': {'lr': 0.01}, 'gamma': 0.1},
         clip_args = {'clip_norm': 5.}
   )
   svi = pyro.infer.SVI(model, guide, scheduler,
         loss=pyro.infer.Trace_ELBO())


   loss = 0.
   for step in range(8000):
      loss += svi.step(data)
      if (step+1) % 500 == 0:
         sys.stdout.write("{}\t{}\t{:.3f}\n".format(iterno+1, step+1, loss / 1e9))
         loss = 0.
      if (step+1) % 6000 == 0:
         scheduler.step()
      if (step+1) % 7000 == 0:
         scheduler.step()
