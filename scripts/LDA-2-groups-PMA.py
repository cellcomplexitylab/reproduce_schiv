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
global N # Number of types.
global G # Number of genes.

# Set the umber of topics now.
K = 2

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


class warmup_scheduler(torch.optim.lr_scheduler.ChainedScheduler):
   def __init__(self, optimizer, warmup=100, decay=9500):
      self.warmup = warmup
      self.decay = decay
      warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.01,
            end_factor = 1.,
            total_iters = self.warmup
      )
      linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1.,
            end_factor = 0.05,
            total_iters=self.decay
      )
      super().__init__([warmup, linear_decay])


def model(data=None):
   # Split data over cell, batch, cell type and expresion.
   batch, ctype, X = data

   # CUDA / CPU and FP16, FP32, FP64 compatibility
   device = X.device
   dtype = X.dtype

   with pyro.plate("genes", G):
      with pyro.plate("batches", M):
         batch_effects = pyro.sample(
               name = "batch_effects",
               fn = dist.Normal(torch.zeros(1).to(device, dtype), .1)
         ).unsqueeze(-3)
      with pyro.plate("types", N):
         type_effects = pyro.sample(
               name = "type_effects",
               fn = dist.Normal(torch.zeros(1).to(device, dtype), .1)
         ).unsqueeze(-3)

   one_over_K = torch.ones(1).to(device, dtype) / K
   one_over_G = torch.ones(G).to(device, dtype) / G

   # Sample globals.
   with pyro.plate("topics", K):
      # Sample global frequencies of the topics.
      topic_weights = pyro.sample(
            name = "alpha",
            fn = dist.Gamma(one_over_K, 1.)
      ).unsqueeze(-2)
      # Sample word frequencies by topic.
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
            fn = dist.Dirichlet(topic_weights),
      )
      # Sampling a topic then a word is equivalent
      # to sampling a word from weighted frequencies.
      freqs = torch.matmul(doc_topics, word_freqs)
      # Apply batch effects.
      batch_mat = batch_effects.exp()
      freqs *= torch.matmul(F.one_hot(batch.to(torch.int64)).to(device, dtype), batch_mat)
      # Apply type effects.
      type_mat = type_effects.exp()
      freqs *= torch.matmul(F.one_hot(ctype.to(torch.int64)).to(device, dtype), type_mat)
      # Sample word counts in document (freqs no long add up to 1).
      data = pyro.sample(
            name = "g",
            fn = dist.Multinomial(probs = freqs, validate_args = False),
            obs = X
      )

   return topic_weights, word_freqs, data


def guide(data=None):
   # Split data over cell, batch, cell type and expresion.
   batch, ctypes, X = data

   # CUDA / CPU and FP16, FP32, FP64 compatibility
   device = X.device
   dtype = X.dtype

   # Use a conjugate guide for global variables.
   topic_weights_posterior = pyro.param(
         "topic_weights_posterior",
         lambda: torch.ones(K).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   word_freqs_posterior = pyro.param(
         "word_freqs_posterior",
         lambda: torch.ones(K, G).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )

   batch_effects_posterior = pyro.param(
         "batch_effects_posterior",
         lambda: torch.ones(M,G).to(device, dtype)
   )
   type_effects_posterior = pyro.param(
         "type_effects_posterior",
         lambda: torch.ones(N,G).to(device, dtype),
   )

   with pyro.plate("genes", G):
      with pyro.plate("batches", M):
         batch_effects = pyro.sample(
               name = "batch_effects",
               fn = dist.Normal(batch_effects_posterior, .02)
         ).unsqueeze(-3)
      with pyro.plate("types", N):
         type_effects = pyro.sample(
               name = "type_effects",
               fn = dist.Normal(type_effects_posterior, .02)
         ).unsqueeze(-3)

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
   doc_topics_posterior = pyro.param(
         "doc_topic_posterior",
         lambda: torch.ones(ndocs,K).to(device, dtype),
         constraint = torch.distributions.constraints.greater_than(0.5)
   )
   with pyro.plate("documents", ndocs):
      pyro.sample(
            name = "theta",
            fn = dist.Dirichlet(doc_topics_posterior)
      )


def sc_data(fname, device="cuda", dtype=torch.float64):
   """ 
   Data for single-cell transcriptome, returns a 4-tuple with
      1. a list of cell identifiers,
      2. a tensor of batches as integers,
      3. a tensor of cell type
      4. a tensor with read counts.
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
   # Return the (cells, batches, types, expressions) tuple.§
   batch_tensor = torch.tensor(list_of_batch_ids).to(device, dtype)
   ctype_tensor = torch.tensor(list_of_ctype_ids).to(device, dtype)
   expr_tensor = torch.stack(list_of_exprs).to(device, dtype)
   return list_of_cells, batch_tensor, ctype_tensor, expr_tensor


data = sc_data('alivecells.tsv')

pyro.clear_param_store()
torch.manual_seed(123)
pyro.set_rng_seed(123)

# Set global dimensions.
cells, batches, ctypes, X = data
M = int(batches.max() + 1)
N = int(ctypes.max() + 1)
G = int(X.shape[-1])
data = batches, ctypes, X

pyro.clear_param_store()

scheduler = pyro.optim.PyroLRScheduler(
      scheduler_constructor = warmup_scheduler,
      optim_args = {
         "optimizer": torch.optim.AdamW,
         "optim_args": {"lr": 0.01}, "warmup": 400, "decay": 9600,
      },
      clip_args = {"clip_norm": 5.}
)


svi = pyro.infer.SVI(
         model = model,
         guide = guide,
         optim = scheduler,
         loss = pyro.infer.JitTrace_ELBO(
            num_particles = 25,
            vectorize_particles = True,
            max_plate_nesting = 2
         )
      )

loss = 0.
for step in range(10000):
   loss += svi.step((batches, ctypes, X))
   scheduler.step()
   if (step+1) % 500 == 0:
      sys.stderr.write("iter {}: loss = {:.2f}\n".format(step+1, loss / 1e9))
      loss = 0.


###
out = pyro.param("doc_topic_posterior")
wfreq = pyro.param("word_freqs_posterior")
batch_effects_posterior = pyro.param("batch_effects_posterior")
type_effects_posterior = pyro.param("type_effects_posterior")
# Output signature breakdown with row names.
pd.DataFrame(out.detach().cpu().numpy(), index=cells) \
   .to_csv("out-PMA-2.txt", sep="\t", header=False, float_format="%.5f")
np.savetxt("wfreq-PMA-2.txt", wfreq.detach().cpu().numpy(), fmt="%.5f")
np.savetxt("batch-effects-PMA-2.txt",
      batch_effects_posterior.detach().cpu().numpy(), fmt="%.5f")
np.savetxt("type-effects-PMA-2.txt",
      type_effects_posterior.detach().cpu().numpy(), fmt="%.5f")
