import numpy as np
import pyro
import pyro.distributions as dist
import re
import random
import sys
import torch
import torch.nn.functional as F

from local import warmup_scheduler, sc_data

global K # Number of signatures.
global G # Number of genes.

pyro.enable_validation(False)


def model(X=None, generate=0):
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ndocs = X.shape[0] if X is not None else generate

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
   with pyro.plate("documents", ndocs):
      # Sample topic frequencies in document.
      doc_topics = pyro.sample(
            name = "theta",
            fn = dist.Dirichlet(topic_weights),
      )
      # Sampling a topic then a word is equivalent
      # to sampling a word from weighted frequencies.
      freqs = torch.matmul(doc_topics, word_freqs)
      # Sample word counts in document. The option 'validate_args'
      # is set to 'False' because the total counts are not equal
      # for every document. This is not a problem to compute log
      # probabilities, but the distribution will not produce valid
      # random samples in this case.
      return pyro.sample(
            name = "g",
            fn = dist.Multinomial(probs = freqs, validate_args = False),
            obs = X
      )


def guide(X=None, generate=0):
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ndocs = X.shape[0] if X is not None else generate

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

   with pyro.plate("topics", K):
      alpha = pyro.sample(
            name = "alpha",
            fn = dist.Gamma(topic_weights_posterior, 1.)
      )
      phi = pyro.sample(
            name = "phi",
            fn = dist.Dirichlet(word_freqs_posterior)
      )

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


if __name__ == "__main__":

   K = int(sys.argv[1])
   in_fname = sys.argv[2]
   out_fname = sys.argv[3]

   # Read in the data and set the dimensions.
   data = sc_data(in_fname)

   cells, batches, ctypes, X = data
   B = int(batches.max() + 1)
   N = int(ctypes.max() + 1)
   G = int(X.shape[-1])
   data = batches, ctypes, X

   scheduler = pyro.optim.PyroLRScheduler(
         scheduler_constructor = warmup_scheduler,
         optim_args = {
            "optimizer": torch.optim.AdamW,
            "optim_args": {"lr": 0.01}, "warmup": 400, "decay": 6000,
         },
         clip_args = {"clip_norm": 5.}
   )

   pyro.clear_param_store()
   svi = pyro.infer.SVI(
      model = model,
      guide = guide,
      optim = scheduler,
      loss = pyro.infer.JitTrace_ELBO(
         num_particles = 16,
         vectorize_particles = True,
         max_plate_nesting = 2
      )
   )

   loss = 0.
   for step in range(6000):
      loss += svi.step(X)
      scheduler.step()
      if (step+1) % 500 == 0:
         sys.stderr.write("iter {}: loss = {:.2f}\n".format(step+1, loss/1e9))
         loss = 0.

   # Model parameters.
   names = ("topic_weights_posterior", "word_freqs_posterior",
         "doc_topic_posterior")
   params = { name: pyro.param(name).detach().cpu() for name in names }

   # Posterior predictive sampling.
   predictive = pyro.infer.Predictive(
         model = model,
         guide = guide,
         num_samples = 1000,
         return_sites = ("theta", "phi"),
   )
   sim = predictive(None, generate=X.shape[0])
   # Resample the full transcriptome for each cell.
   # This must be done manually because sampling from
   # the multinomial is faulty when the counts are
   # not constant.
   freqs = torch.matmul(sim["theta"], sim["phi"]).to("cpu")
   total_counts = X.sum(dim=1)
   smpl = list()
   for i in range(X.shape[0]):
      smpl_i = list()
      multinom = torch.distributions.Multinomial(
            total_count = int(total_counts[i]),
            probs = freqs[:,i,:],
            validate_args = True
      )
      with torch.no_grad():
         smpl.append(multinom.sample([1]))
   tx = torch.stack(smpl).transpose(0,1)
   smpl = { "tx": tx.detach().cpu() }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
