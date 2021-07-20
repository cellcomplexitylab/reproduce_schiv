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
K = 3


def model(data=None):
   # Split data over cell, batch, cell type and expresion.
   cells, batch, ctype, X = data

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
         list_of_cells.append(cell)
         list_of_infos.append(info)
         list_of_exprs.append(torch.tensor(expr))
   # Extract batch (plate) from cell ID.
   list_of_plates = [re.sub(r"_.*", "", x) for x in list_of_cells]
   unique_plates = list(set(list_of_plates))
   list_of_batch_ids = [unique_plates.index(x) for x in list_of_plates]
   # Extract cell type from treatment info.
   list_of_ctypes = [re.sub(r"\+.*", "", x) for x in list_of_infos]
   unique_ctypes = list(set(list_of_ctypes))
   list_of_ctype_ids = [unique_ctypes.index(x) for x in list_of_ctypes]
   # Return the (cells, batches, types, expressions) tuple.
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
# Output signature breakdown with row names.
pd.DataFrame(out.detach().cpu().numpy(), index=cells) \
   .to_csv("out-alive-no-bec-no-HIV.txt", sep="\t", header=False, float_format="%.5f")
np.savetxt("wfreq-alive-no-bec-no-HIV.txt", wfreq.detach().cpu().numpy(), fmt="%.5f")
