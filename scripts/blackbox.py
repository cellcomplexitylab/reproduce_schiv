import numpy as np
import pyro
import pyro.distributions as dist
import re
import random
import sys
import torch
import torch.nn.functional as F

pyro.enable_validation(False)

global K # Number of modules.
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


def model(data, generate=0):

   batch, ctype, X = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # dim: ncells x B
   one_hot_batch = F.one_hot(batch.to(torch.int64)).to(device, dtype)

   # Zero-inflation factor, 'pi'. The median is set
   # at 0.15, with 5% that 'pi' is less than 0.01 and
   # 5% that 'pi' is more than 50%.
   pi = pyro.sample(
         # dim: 1 x 1 x 1 | .
         name = "pi",
         fn = dist.Beta(
            1. * torch.ones(1,1,1).to(device, dtype),
            4. * torch.ones(1,1,1).to(device, dtype)
         )
   )

   # Variance-to-mean model. Variance is modelled as
   # 's * u', where 'u' is mean gene expression and
   # 's' is a trained parameters with 90% chance of
   # being in the interval (1.5,15).
   s = 1. + pyro.sample(
         name = "s",
         # dim: 1 x 1 x 1 | .
         fn = dist.LogNormal(
            1.0 * torch.ones(1,1,1).to(device, dtype),
            0.5 * torch.ones(1,1,1).to(device, dtype)
         )
   )

   with pyro.plate("G", G):

      # Batch effects (mixed) on every gene. The 'loc'
      # hyper parameter has 90% chance of being in the
      # interval (-4,6), i.e., from 0 to 400 reads.
      # The 'scale' hyper parameter has 90% chance of
      # being below XXX, meaning that batch effects
      # are expected to be up to 50% for 99% of genes.
      h1_b_loc = pyro.sample(
            name = "h1_b_loc",
            # dim: 1 x 1 x G | .
            fn = dist.Normal(
               1 * torch.ones(1,1,1).to(device, dtype),
               3 * torch.ones(1,1,1).to(device, dtype)
            )
      )
      h1_b_scale = pyro.sample(
            name = "h1_b_scale",
            # dim: 1 x 1 x G | .
            fn = dist.HalfNormal(
               0.1 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # Gene modules (mixed).
      h2_g_scale = pyro.sample(
            name = "h2_g_scale",
            # dim: 1 x 1 x G | .
            fn = dist.HalfNormal(
               0.25 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      with pyro.plate("Bx1xG", B):

         # Batch effect (mixed). See hyper parameters.
         bmat = pyro.sample(
               name = "bmat",
               # dim: 1 x B x G | .
               fn = dist.Normal(
                  h1_b_loc,  # dim 1 x 1 x G
                  h1_b_scale # dim 1 x 1 x G
               )
         )
         # Assign baselines (multiply 'bmat' with a one-hot
         # encoding of the batches). 
         # dim: ncells x G
         baselines = torch.einsum("ai,...xib->...ab", one_hot_batch, bmat)


      with pyro.plate("KxG", K):

         # Gene modules (mixed). See hyper parameters.
         h1_g_loc = pyro.sample(
               name = "h1_g_loc",
               # dim: 1 x K x G | .
               fn = dist.Normal(
                  torch.zeros(1,1,1).to(device, dtype),
                  h2_g_scale # dim: 1 x 1 x G
               )
         )

         with pyro.plate("BxKxG", B):

            # Gene modules (mixed). See hyper parameters.
            gmat = pyro.sample(
                  name = "gmat",
                  # dim: B x K x G | .
                  fn = dist.Normal(
                     h1_g_loc, # dim: 1 x K x G
                     0.01 * torch.ones(1,1,1).to(device, dtype),
                  )
            )
            # Assign signatures (multiply 'gmat' with a
            # one-hot encoding of the batches).
            # dim: K x ncells x G
            sig = torch.einsum("ai,...ibc->...bac", one_hot_batch, gmat)

   with pyro.plate("K", K):
      # Expected proportion of modules, 'alpha'. This
      # is the usual prior for topic modeling. It is
      # symmetric with average 1/K and favors strong-or-
      # completely-absent types of splits.
      alpha = pyro.sample(
            name = "alpha",
            # dim: 1 x 1 x K | .
            fn = dist.Gamma(torch.ones(1,1,K).to(device, dtype) / K, 1)
      )

   with pyro.plate("ncells", ncells):
      # Sample module frequencies in cells. This is
      # the usual sampling for topic modeling.
      theta = pyro.sample(
            name = "theta",
            # dim: 1 x 1 x ncells | K
            fn = dist.Dirichlet(alpha.unsqueeze(-4))
      )

   # ----------------------------------------------------------------
   # A Dirichlet event is a vector, so the rightmost dimension of the
   # output (with size K) is not considered to be part of the batch.
   # If a dimension is not added to 'alpha', this shifts the batch
   # shape and causes a mismatch between the model and the guide.
   # ----------------------------------------------------------------

      # dim: ncells x G
      tuning = torch.einsum("...xyai,...iab->...ab", theta, sig)

      # Relative read count per cell, 'shift'. The prior is
      # chosen so that the median is 1 with a 5% chance that 'c'
      # is less than 0.5 and a 5% chance that it is more than 5.
      shift = pyro.sample(
            name = "shift",
            # dim: 1 x 1 x ncells | .
            fn = dist.Normal(
               0. * torch.zeros(1,1,1).to(device, dtype),
               1. * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # Change the dimension to dim: ncells x 1.
      shift = shift.squeeze().unsqueeze(-1)

   # ----------------------------------------------------------------
   # This seems odd, right? Two dimensions are added on the left of
   # 'ncells' to make it explicit that 'pyro.sample()' adds dummy
   # dimensions to every tensor. Implicit batching like particles is
   # added to the left. Squeezing pops the two singleton dimensions
   # and unsqueezing adds a dimension to the right. The result is
   # congruent whether implicit batching is added or not.
   # ----------------------------------------------------------------

      # dim: 1 x ncells x G
      u = torch.exp(shift + baselines + tuning).unsqueeze(-3)

      # Variance and parameters of the negative binomial.
      # Parameter 'u' is the average number of reads and
      # the variance is 's x u'. Parametrize 'r' and 'p'
      # as a function of 'u' and 's'. Parameter 'u' has
      # dimensions ncells x G and 's' has dimension 1
      # (the result has dimension ncells x G).
      p_ = 1. - 1. / s # dim:          1
      r_ = u / (s - 1) # dim: ncells x G

   # ----------------------------------------------------------------
   # If the variance is assumed to vary with a power 'a', i.e. the
   # variance is 's x u^a', then the equations above become:
   # p_ = 1. - 1. / (s*u^(a-1))
   # r_ = u / (s*u^(a-1) - 1)
   # ----------------------------------------------------------------

      # Make sure that parameters of the ZINB are valid.
      eps = 1e-6
      p = torch.clamp(p_, min=0.+eps, max=1.-eps)
      r = torch.clamp(r_, min=0.+eps)

      # Observations are sampled from a ZINB distribution.
      return pyro.sample(
            name = "X",
            # dim: ncells | G
            fn = dist.ZeroInflatedNegativeBinomial(
               total_count = r, # dim: ncells x G
               probs = p,       # dim:          1
               gate = pi        # dim:          1
            ).to_event(1),
            obs = X
      )


def guide(data=None, generate=0):

   batch, ctype, X = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # Posterior distribution of 'pi'.
   posterior_pi_0 = pyro.param(
         "posterior_pi_0", # dim: 1 x 1 x 1
         lambda: 1. * torch.ones(1,1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   posterior_pi_1 = pyro.param(
         "posterior_pi_1", # dim: 1 x 1 x 1
         lambda: 4. * torch.ones(1,1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   posterior_pi = pyro.sample(
         name = "pi",
         # dim: 1 x 1 x 1 | .
         fn = dist.Beta(
            posterior_pi_0, # dim: 1 x 1 x 1
            posterior_pi_1  # dim: 1 x 1 x 1
         )
   )

   # Posterior distribution of 's'.
   posterior_s_loc = pyro.param(
         "posterior_s_loc", # dim: 1 x 1 x 1
         lambda: torch.ones(1,1,1).to(device, dtype)
   )
   posterior_s_scale = pyro.param(
         "posterior_s_scale", # dim: 1 x 1 x 1
         lambda: torch.ones(1,1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   pyro.sample(
      name = "s",
      # dim: 1 x 1 x 1 | .
      fn = dist.LogNormal(
         posterior_s_loc,  # dim: 1 x 1 x 1
         posterior_s_scale # dim: 1 x 1 x 1
      )
   )

   with pyro.plate("G", G):

      # Posterior distribution of 'h1_b_loc'.
      posterior_h1_b_loc_loc = pyro.param(
            "posterior_h1_b_loc_loc",
            lambda: 1 * torch.ones(1,1,G).to(device, dtype)
      )
      posterior_h1_b_loc_scale = pyro.param(
            "posterior_h1_b_loc_scale",
            lambda: 3 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      h1_b_loc = pyro.sample(
            name = "h1_b_loc",
            # dim: 1 x 1 x G | .
            fn = dist.Normal(
               posterior_h1_b_loc_loc,  # dim: 1 x 1 x G
               posterior_h1_b_loc_scale # dim: 1 x 1 x G
            )
      )

      # Posterior distribution of 'h1_b_scale'.
      posterior_h1_b_scale_loc = pyro.param(
            "posterior_h1_b_scale_loc",
            lambda: -2.7 * torch.ones(1,G).to(device, dtype)
      )
      posterior_h1_b_scale_scale = pyro.param(
            "posterior_h1_b_scale_scale",
            lambda: 0.6 * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      h1_b_scale = pyro.sample(
            name = "h1_b_scale",
            # dim: 1 x 1 x G | .
            fn = dist.LogNormal(
               posterior_h1_b_scale_loc,  # dim: 1 x 1 x G
               posterior_h1_b_scale_scale # dim: 1 x 1 x G
            )
      )

      # Posterior distribution of 'h2_g_scale'.
      posterior_h2_g_scale_loc = pyro.param(
            "posterior_h2_g_scale_loc",
            lambda: -1.8 * torch.ones(1,1,G).to(device, dtype)
      )
      posterior_h2_g_scale_scale = pyro.param(
            "posterior_h2_g_scale_scale",
            lambda: 0.5 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      h2_g_scale = pyro.sample(
            name = "h2_g_scale",
            # dim: 1 x 1 x G | .
            fn = dist.LogNormal(
               posterior_h2_g_scale_loc,  # dim: 1 x 1 x G
               posterior_h2_g_scale_scale # dim: 1 x 1 x G
            )
      )

      with pyro.plate("Bx1xG", B):

        # Posterior distribution of 'bmat'.
        posterior_b_loc = pyro.param(
               "posterior_b_loc",
               lambda: torch.ones(1,B,G).to(device, dtype)
        )
        posterior_b_scale = pyro.param(
               "posterior_b_scale",
               lambda: torch.ones(1,B,G).to(device, dtype),
               constraint = torch.distributions.constraints.positive
        )
        pyro.sample(
               name = "bmat",
               # dim: 1 x B x G | .
               fn = dist.Normal(
                  posterior_b_loc,  # dim: 1 x B x G
                  posterior_b_scale # dim: 1 x B x G
               )
        )

      with pyro.plate("KxG", K):

         # Posterior distribution of 'h1_g_loc'.
         posterior_h1_g_loc_loc = pyro.param(
               "posterior_h1_g_loc_loc",
               lambda: 0 * torch.zeros(1,K,G).to(device, dtype)
         )
         posterior_h1_g_loc_scale = pyro.param(
               "posterior_h1_g_loc_scale",
               lambda: 0.5 * torch.ones(1,K,G).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )
         h1_g_loc = pyro.sample(
               name = "h1_g_loc",
               # dim: 1 x K x G | .
               fn = dist.Normal(
                  posterior_h1_g_loc_loc,  # dim: 1 x K x G
                  posterior_h1_g_loc_scale # dim: 1 x K x G
               )
         )

         with pyro.plate("BxKxG", B):

            # Posterior distribution of 'gmat'.
            posterior_g_loc = pyro.param(
                  "posterior_g_loc",
                  lambda: torch.zeros(B,K,G).to(device, dtype),
            )
            posterior_g_scale = pyro.param(
                  "posterior_g_scale",
                  lambda: 0.03 * torch.ones(B,K,G).to(device, dtype),
                  constraint = torch.distributions.constraints.positive
            )
            pyro.sample(
                  name = "gmat",
                  # dim: B x K x G | .
                  fn = dist.Normal(
                     posterior_g_loc,  # dim: B x K x G
                     posterior_g_scale # dim: B x K x G
                  )
            )

   with pyro.plate("K", K):
      # Posterior distribution of 'alpha'.
      posterior_alpha = pyro.param(
            "posterior_alpha",
            lambda: torch.ones(1,1,K).to(device, dtype) / K,
            constraint = torch.distributions.constraints.positive
      )
      alpha = pyro.sample(
            name = "alpha",
            # dim: 1 x 1 x K | .
            fn = dist.Gamma(posterior_alpha, 1)
      )

   with pyro.plate("ncells", ncells):
      # Posterior distribution of 'theta'.
      posterior_theta = pyro.param(
            "posterior_theta",
            lambda: torch.ones(1,1,ncells,K).to(device, dtype),
            constraint = torch.distributions.constraints.greater_than(0.5)
      )
      theta = pyro.sample(
            name = "theta",
            # dim: 1 x 1 x ncells | K
            fn = dist.Dirichlet(posterior_theta)
      )

      # Posterior distribution of 'shift'.
      posterior_c_loc = pyro.param(
            "posterior_c_loc",
            lambda: 1. * torch.zeros(1,ncells).to(device, dtype),
      )
      posterior_c_scale = pyro.param(
            "posterior_c_scale",
            lambda: 1. * torch.ones(1,ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "shift",
            # dim: ncells
            fn = dist.Normal(
               posterior_c_loc,  # dim: 1 x ncells
               posterior_c_scale # dim: 1 x ncells
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
   # Parsing function.
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
            "optim_args": {"lr": 0.01}, "warmup": 400, "decay": 4000,
         },
         clip_args = {"clip_norm": 5.}
   )

   pyro.clear_param_store()
   svi = pyro.infer.SVI(
      model = model,
      guide = guide,
      optim = scheduler,
      loss = pyro.infer.JitTrace_ELBO(
         num_particles = 4,
         vectorize_particles = True,
         max_plate_nesting = 3
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
   for step in range(4000):
      loss += svi.step(data)
      scheduler.step()
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,3)}\n")
         loss = 0.

   # Model parameters.
   names = (
      "posterior_pi_0", "posterior_pi_1",
      "posterior_s_loc", "posterior_s_scale",
      "posterior_b_loc", "posterior_b_scale",
      "posterior_c_loc", "posterior_c_scale",
      "posterior_g_loc", "posterior_g_scale",
      "posterior_alpha", "posterior_theta",
      "posterior_h1_b_loc_loc", "posterior_h1_b_loc_scale",
      "posterior_h1_b_scale_loc", "posterior_h1_b_scale_scale",
      "posterior_h2_g_scale_loc", "posterior_h2_g_scale_scale",
      "posterior_h1_g_loc_loc", "posterior_h1_g_loc_scale",
   )
   params = { name: pyro.param(name).detach().cpu() for name in names }
   torch.save({"params":params}, out_fname)

   # Posterior predictive sampling.
   predictive = pyro.infer.Predictive(
         model = model,
         guide = guide,
         num_samples = 1000,
         return_sites = ("theta", "_RETURN"),
   )
   sim = predictive(data=(batches, ctypes, None), generate=X.shape[0])
   # Resample the full transcriptome for each cell.
   smpl = {
      "tx": sim["_RETURN"].detach().cpu(),
      "theta": sim["theta"].detach().cpu(),
   }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
