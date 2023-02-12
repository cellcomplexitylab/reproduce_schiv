import numpy as np
import pyro
import pyro.distributions as dist
import re
import random
import sys
import torch
import torch.nn.functional as F

pyro.enable_validation(False)

global K # Number of signatures.
global B # Number of batches.
global N # Number of types
global G # Number of genes.


class warmup_scheduler(torch.optim.lr_scheduler.ChainedScheduler):
   def __init__(self, optimizer, warmup=100, decay=4500):
      self.warmup = warmup
      self.decay = decay
      warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.,
            total_iters=self.warmup
      )
      linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.,
            end_factor=0.05,
            total_iters=self.decay
      )
      super().__init__([warmup, linear_decay])


def model(data=None, generate=0):
   if data is None:
      batch = ctype = X = None
   else:
      batch, ctype, X = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # Zero-inflation factor, 'pi'. The prior is chosen so that
   # there is a 5% chance that 'pi' is greater than 0.1.
   pi = pyro.sample(
         # dim: 1 | .
         name = "pi",
         fn = dist.Beta(
            1. * torch.ones(1).to(device, dtype),
            25. * torch.ones(1).to(device, dtype)
         )
   )

   with pyro.plate("K", K):
      # Expected proportion of signatures, 'alpha'. This is the
      # usual prior for topic modeling. It is symmetric with
      # average 1/K and favors strong-or-completely-absent types
      # of splits.
      alpha = pyro.sample(
            name = "alpha",
            # dim: K | .
            fn = dist.Gamma(torch.ones(K).to(device, dtype) / K, 1)
      ).unsqueeze(-2)
   
   # --------------------------------------------------------------
   # The 'unsqueeze()' statement is essential. It allows the output
   # to have one extra dimension "left" of the event.
   # --------------------------------------------------------------
   
      with pyro.plate("GxK", G):
         # Departure expression from the baseline (see below).
         # The 5-th and 95-th quantiles are 0.4 and 2.3, meaning
         # that about 5% of the genes have a 5-fold difference
         # between signatures. The 1-st and 99-th quantiles are 0.3
         # and 3.2, or a 10-fold difference for 1% of the genes.
         g = pyro.sample(
            name = "g",
            # dim: G x K | .
            fn = dist.LogNormal(
               .0 * torch.zeros(G,K).to(device, dtype),
               .5 * torch.ones(G,K).to(device, dtype)
            )
         )

   with pyro.plate("G", G):
      # Baseline expected number of reads per gene. With about
      # 10,000 genes and 1,000,000 reads, the average should be
      # around 100 reads per gene. Here the average is ~ 90,
      # the median is 1, the 95-th quantile is ~ 140 and the
      # 99-th quantile is ~ 1000.
      baseline = pyro.sample(
         name = "baseline",
         # dim: 1 x G | .
         fn = dist.LogNormal(
            0. * torch.ones(1,G).to(device, dtype),
            3. * torch.ones(1,G).to(device, dtype)
         )
      )
      # Variance-to-mean ratio per gene, 's'.
      s = 1. + pyro.sample(
         name = "s",
         # dim: 1 x G | .
         fn = dist.LogNormal(
            3. * torch.ones(1,G).to(device, dtype),
            1. * torch.ones(1,G).to(device, dtype)
         )
      )

      if batch is None:
         bfx = 1.
      else:
         with pyro.plate("BxG", B):
            # Batch effect.
            bmat = pyro.sample(
               name = "bmat",
               # dim: B x G | .
               fn = dist.LogNormal(
                  .0 * torch.zeros(1,1).to(device, dtype),
                  .2 * torch.ones(1,1).to(device, dtype)
               )
            )
            one_hot = F.one_hot(batch.to(torch.int64)).to(device, dtype)
            bfx = torch.matmul(one_hot, bmat) # dim: ncells x G

      if ctype is None:
         cfx = 1.
      else:
         with pyro.plate("NxG", N):
            # Cell type effect.
            cmat = pyro.sample(
               name = "cmat",
               # dim: N x G | .
               fn = dist.LogNormal(
                  .0 * torch.zeros(1,1).to(device, dtype),
                  .4 * torch.ones(1,1).to(device, dtype)
               )
            )
            one_hot = F.one_hot(ctype.to(torch.int64)).to(device, dtype)
            cfx = torch.matmul(one_hot, cmat) # dim: ncells x G

   with pyro.plate("cells", ncells):
      # Sample signature frequencies in cells.
      theta = pyro.sample(
            name = "theta",
            # dim: ncells x 1 | K
            fn = dist.Dirichlet(alpha)
      )

   # ----------------------------------------------------------------
   # A Dirichlet event is a vector, so the rightmost dimension of the
   # output (with size K) is not considered to be part of the batch.
   # If a dimension is not added to 'alpha', this shifts the batch
   # shape and causes a mismatch between the model and the guide.
   # ----------------------------------------------------------------

      # Normalized expected number of reads in each cell, after
      # applying signatures in proportion (dim: ncells x G).
      lmbd = torch.matmul(theta, baseline * g.transpose(-1,-2))

      # Relative read count per cell, 'c'. The prior is chosen
      # so that the median is 1 with a 5% chance that 'c' is
      # less than 0.5 and a 5% chance that it is more than 5.
      c = pyro.sample(
             name = "c",
             # dim: ncells | .
             fn = dist.LogNormal(
                0. * torch.zeros(1).to(device, dtype),
                1. * torch.ones(1).to(device, dtype)
             )
      )

      # Absolute average read count per cell per gene.
      u = bfx * cfx * lmbd * c.unsqueeze(-1) # dim: ncells x G

      # Variance and parameters of the negative binomial.
      # Parameter 'u' is the average number of reads and
      # the variance is 's' x 'u'. Parametrize 'r' and
      # 'p' as a function of 'u' and 's'. Parameters 'u'
      # and 's' have dimensions ncells x G and 1 x G,
      # respectively. As a consequence, parameters 'r'
      # and 'p' have dimension ncells x G.
      r = u/(s-1)
      p = (s-1)/s

      # Observations are assumed to follow the zero-inflated
      # negative binomial distribution.
      return pyro.sample(
         name = "X",
         # dim: ncells x G | .
         fn = dist.ZeroInflatedNegativeBinomial(
            total_count = r,
            probs = p,
            gate = pi
        ).to_event(1),
         obs = X
      )


def guide(data=None, generate=0):
   if data is None:
      batch = ctype = X = None
   else:
      batch, ctype, X = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # Posterior distribution of 'pi'.
   posterior_pi_0 = pyro.param(
         "posterior_pi_0",
         lambda: 1. * torch.ones(1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   posterior_pi_1 = pyro.param(
         "posterior_pi_1",
         lambda: 4. * torch.ones(1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   posterior_pi = pyro.sample(
         name = "pi",
         fn = dist.Beta(
            posterior_pi_0,
            posterior_pi_1
         )
   )

   with pyro.plate("G", G):
      # For the posterior distribution of 'g' and 'baseline',
      # we want a single scale parameter for all the signatures,
      # in order to prevent the appearance of "fuzzy" signatures.
      posterior_baseline = pyro.param(
            "posterior_baseline",
            lambda: torch.zeros(1,G).to(device, dtype),
      )
      posterior_g_scale = pyro.param(
            "posterior_g_scale",
            lambda: 1 * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      # Posterior distribution of 'baseline'.
      baseline = pyro.sample(
         name = "baseline",
         # dim: 1 x G | .
         fn = dist.LogNormal(
            posterior_baseline,
            posterior_g_scale
         )
      )
      # Posterior distribution of 's'.
      posterior_s_loc = pyro.param(
            "posterior_s_loc",
            lambda: 3. * torch.ones(1,G).to(device, dtype)
      )
      posterior_s_scale = pyro.param(
            "posterior_s_scale",
            lambda: 2. * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      s = pyro.sample(
         name = "s",
         # dim: 1 x G | .
         fn = dist.LogNormal(
            posterior_s_loc,
            posterior_s_scale
         )
      )

      # Uncertainty around 'b' and 't'.
      posterior_bt_scale = pyro.param(
            "posterior_bt_scale",
            lambda: torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )

      if batch is not None:
         with pyro.plate("BxG", B):
            # Posterior distribution of 'b'.
            posterior_b_loc = pyro.param(
                  "posterior_b_loc",
                  lambda: torch.zeros(B,G).to(device, dtype)
            )
            bmat = pyro.sample(
               name = "bmat",
               # dim: B x G | .
               fn = dist.LogNormal(
                  posterior_b_loc,
                  posterior_bt_scale
               )
            )

      if ctype is not None:
         with pyro.plate("NxG", N):
            # Posterior distribution of 't'.
            posterior_t_loc = pyro.param(
                  "posterior_t_loc",
                  lambda: torch.zeros(N,G).to(device, dtype)
            )
            cmat = pyro.sample(
               name = "cmat",
               # dim: N x G | .
               fn = dist.LogNormal(
                  posterior_t_loc,
                  posterior_bt_scale
               )
            )

   with pyro.plate("K", K):
      # Posterior distribution of 'alpha'.
      posterior_alpha = pyro.param(
            "posterior_alpha",
            lambda: torch.ones(K).to(device, dtype) / K,
            constraint = torch.distributions.constraints.positive
      )
      alpha = pyro.sample(
            name = "alpha",
            # dim: K | .
            fn = dist.Gamma(posterior_alpha, 1)
      )
      with pyro.plate("GxK", G):
         # Posterior distribution of 'g'.
         posterior_g_loc = pyro.param(
               "posterior_g_loc",
               lambda: 3.9 * torch.ones(G,K).to(device, dtype),
         )
         g = pyro.sample(
               name = "g",
               # dim: G x K | .
               fn = dist.LogNormal(
                  posterior_g_loc,
                  posterior_g_scale.transpose(-1,-2)
               )
         )

   with pyro.plate("cells", ncells):
      # Posterior distribution of 'theta'.
      posterior_theta = pyro.param(
            "posterior_theta",
            lambda: torch.ones(ncells,K).to(device, dtype),
            constraint = torch.distributions.constraints.greater_than(0.5)
      )
      theta = pyro.sample(
            name = "theta",
            # dim: 1 x ncells | K
            fn = dist.Dirichlet(posterior_theta)
      )

      # Posterior distribution of 'c'.
      posterior_c_loc = pyro.param(
            "posterior_c_loc",
            lambda: 0. * torch.zeros(ncells).to(device, dtype),
      )
      posterior_c_scale = pyro.param(
            "posterior_c_scale",
            lambda: 1. * torch.ones(ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      c = pyro.sample(
            name = "c",
            # dim: ncells
            fn = dist.LogNormal(
               posterior_c_loc,
               posterior_c_scale
            )
      )

def sc_data(fname, device="cuda", dtype=torch.float64):
   """ 
   Data for single-cell transcriptome, returns a 3-tuple with
      1. a list of cell identifiers,
      2. a tensor of batches as integers,
      3. a tensor of cell types as integers,
      4. a tensor with read counts.
   """
   list_of_cells = list()
   list_of_infos = list()
   list_of_exprs = list()
   # Convenience parsing function.
   parse = lambda row: (row[0], row[1], [round(float(x)) for x in row[2:]])
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
   unique_ctypes = sorted(list(set(list_of_ctypes)))
   list_of_ctype_ids = [unique_ctypes.index(x) for x in list_of_ctypes]
   batch_tensor = torch.tensor(list_of_batch_ids).to(device, dtype)
   ctype_tensor = torch.tensor(list_of_ctype_ids).to(device)
   expr_tensor = torch.stack(list_of_exprs).to(device, dtype)
   return list_of_cells, batch_tensor, ctype_tensor, expr_tensor

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
            "optim_args": {"lr": 0.01}, "warmup": 400, "decay": 4500,
         },
         clip_args = {"clip_norm": 5.}
   )

   pyro.clear_param_store()
   svi = pyro.infer.SVI(
      model = model,
      guide = guide,
      optim = scheduler,
      loss = pyro.infer.JitTrace_ELBO(
         num_particles = 10,
         vectorize_particles = True,
         max_plate_nesting = 2
      )
   )

   # -----------------------------------------------------------------
   # Setting 'max_plate_nesting' to 2 allows Pyro to add the particle
   # dimension as the third from the right for every sample, even
   # those that are not within a Pyro plate. With this option, 'alpha'
   # has dimension P x 1 x K, updated to P x 1 x 1 x K after the
   # 'unsqueeze()' statement. Then 'theta' is sampled as a Dirichlet
   # with batch shape P x 1 x 1 and event shape K. Because 'theta'
   # is in a plate of size 'ncells', the batch shape is updated to
   # P x 1 x 'ncells'. In the guide, 'theta' is sampled as a Dirichlet
   # from a tensor with shape 'ncells' x K, i.e. with batch shape
   # 'ncells' and event shape K. The sample is in a plate of size
   # 'ncells', which does not affect the sample shape but the option
   # 'max_plate_nesting = 2' makes it 1 x 'ncells' and the particles
   # make it P x 1 x 'ncells' as in the model.
   # -----------------------------------------------------------------

   loss = 0.
   for step in range(4500):
      loss += svi.step(data)
      scheduler.step()
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,2)}\n")
         loss = 0.

   # Model parameters.
   names = (
      "posterior_pi_0", "posterior_pi_1",
      "posterior_s_loc", "posterior_s_scale",
      "posterior_b_loc", "posterior_t_loc", "posterior_bt_scale",
      "posterior_baseline", "posterior_g_loc", "posterior_g_scale",
      "posterior_c_loc", "posterior_c_scale",
      "posterior_alpha", "posterior_theta")
   params = { name: pyro.param(name).detach().cpu() for name in names }

   # Posterior predictive sampling.
   predictive = pyro.infer.Predictive(
         model = model,
         guide = guide,
         num_samples = 1000,
         return_sites = ("theta", "g", "_RETURN"),
   )
   sim = predictive(data=(batches, ctypes, None), generate=X.shape[0])
   # Resample the full transcriptome for each cell.
   tx = sim["_RETURN"]
   # Regenerate 'lmbd' from 'theta' and 'g'. This is the relative
   # gene expression so it represents the inferred average
   # expression of each gene in a given cell.
   lmbd = torch.bmm(sim["theta"].squeeze(), sim["g"].transpose(-1,-2))
   smpl = { "tx": tx.detach().cpu(), "lmbd": lmbd.detach().cpu() }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
