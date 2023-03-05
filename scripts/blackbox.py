import numpy as np
import pyro
import pyro.distributions as dist
import re
import random
import sys
import torch
import torch.nn.functional as F

global K # Number of modules.
global B # Number of batches.
global N # Number of types
global G # Number of genes.

pyro.enable_validation(False)

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

   # dim(one_hot_b): ncells x B
   one_hot_b = F.one_hot(batch.to(torch.int64)).to(device, dtype)
   # dim(one_hot_c): ncells x N
   one_hot_c = F.one_hot(ctype.to(torch.int64)).to(device, dtype)

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

      # Base expression (mixed) on every gene. The
      # base has 90% chance of being in the interval
      # (-4,6), i.e., from 0 to 400 reads (many
      # adjustments will apply, see below).
      base = pyro.sample(
            name = "base",
            # dim(base): 1 x 1 x G | .
            fn = dist.Normal(
               1 * torch.ones(1,1,1).to(device, dtype),
               3 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # Variation of expression over cell types.
      base_dev_over_N = pyro.sample(
            name = "base_dev_over_N",
            # dim(base_dev_over_N): 1 x 1 x G | .
            fn = dist.HalfNormal(
               .25 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # Variation of expression over batches.
      base_dev_over_B = pyro.sample(
            name = "base_dev_over_B",
            # dim(base_dev_over_B): 1 x 1 x G | .
            fn = dist.HalfNormal(
               .1 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # Variation of signatures over cell types.
      sig_dev_over_K = pyro.sample(
            name = "sig_dev_over_K",
            # dim(sig_dev_over_K): 1 x 1 x G | .
            fn = dist.HalfNormal(
               .25 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # Variation of signature over cell types.
      sig_dev_over_N = pyro.sample(
            name = "sig_dev_over_N",
            # dim(sig_dev_over_N): 1 x 1 x G | .
            fn = dist.HalfNormal(
               .1 * torch.ones(1,1,1).to(device, dtype)
            )
      )

      with pyro.plate("KxG", K):

         # Signatures (cell-type-less).
         sig_K = pyro.sample(
               name = "sig_K",
               # dim(sig_K): 1 x K x G | .
               fn = dist.Normal(
                  .0 * torch.zeros(1,1,1).to(device, dtype),
                  sig_dev_over_K
               )
         )

         with pyro.plate("NxKxG", N):

            sig_KN = pyro.sample(
                  name = "sig_KN",
                  # dim(sig_KN): N x K x G | .
                  fn = dist.Normal(
                     sig_K,         # dim: 1 x K x G
                     sig_dev_over_N # dim: 1 x 1 x G
                  )
            )

      with pyro.plate("NxG", N):

         # Generate baselines.
         base_N = pyro.sample(
               name = "base_N",
               # dim(base_N): 1 x N x G | .
               fn = dist.Normal(
                  base,           # dim: 1 x 1 x G
                  base_dev_over_N # dim: 1 x 1 x G
               )
         )

      with pyro.plate("BxG", B):

         base_B = pyro.sample(
               name = "base_B",
               # dim(base_B): 1 x B x G | .
               fn = dist.Normal(
                  torch.zeros(1,1,1).to(device, dtype),
                  base_dev_over_B # dim: 1 x 1 x G
               )
         )

      # dim(base_n): ncells x G
      base_n = torch.einsum("ai,...xib->...ab", one_hot_c, base_N) + \
               torch.einsum("ai,...xib->...ab", one_hot_b, base_B)

   with pyro.plate("ncells", ncells):

      # Sample module frequencies in cells. This is
      # the usual sampling for topic modeling.
      theta = pyro.sample(
            name = "theta",
            # dim(theta): 1 x K x ncells | .
            fn = dist.Normal(
               0 * torch.zeros(1,K,1).to(device, dtype),
               1 * torch.ones(1,K,1).to(device, dtype)
            )
      )

      # dim(tuning): ncells x N x G
      nNG = torch.einsum("...xia,...bic->...abc", theta, sig_KN)

      # dim(tuning): ncells x G
      tuning = torch.einsum("ai,...aib->...ab", one_hot_c, nNG)

      # Relative read count per cell, 'shift'. The prior is
      # chosen so that the median is 1 with a 5% chance that 'c'
      # is less than 0.5 and a 5% chance that it is more than 5.
      shift = pyro.sample(
            name = "shift",
            # dim(shift): 1 x ncells | .
            fn = dist.Normal(
               0. * torch.zeros(1,1,1).to(device, dtype),
               1. * torch.ones(1,1,1).to(device, dtype)
            )
      )

      # dim(shift): ncells x 1.
      shift = shift.squeeze().unsqueeze(-1)

   # ----------------------------------------------------------------
   # This seems odd, right? Two dimensions are added on the left of
   # 'ncells' to make it explicit that 'pyro.sample()' adds dummy
   # dimensions to every tensor. Implicit batching like particles is
   # added to the left. Squeezing pops the two singleton dimensions
   # and unsqueezing adds a dimension to the right. The result is
   # congruent whether implicit batching is added or not.
   # ----------------------------------------------------------------

      # dim(u): ncells x G
      u = torch.exp(base_n + shift + tuning)

      # Variance and parameters of the negative binomial.
      # Parameter 'u' is the average number of reads and
      # the variance is 's x u'. Parametrize 'r' and 'p'
      # as a function of 'u' and 's'. Parameter 'u' has
      # dimensions ncells x G and 's' has dimension 1
      # (the result has dimension ncells x G).
      p_ = 1. - 1. / s # dim(p_): 1
      r_ = u / (s - 1) # dim(r_): ncells x G

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
            # dim(X): ncells | G
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
   post_pi_0 = pyro.param(
         "post_pi_0", # dim: 1 x 1 x 1
         lambda: 1. * torch.ones(1,1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_pi_1 = pyro.param(
         "post_pi_1", # dim: 1 x 1 x 1
         lambda: 4. * torch.ones(1,1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   pyro.sample(
         name = "pi",
         # dim: 1 x 1 x 1 | .
         fn = dist.Beta(
            post_pi_0, # dim: 1 x 1 x 1
            post_pi_1  # dim: 1 x 1 x 1
         )
   )

   # Posterior distribution of 's'.
   post_s_loc = pyro.param(
         "post_s_loc", # dim: 1 x 1 x 1
         lambda: 1 * torch.ones(1,1,1).to(device, dtype)
   )
   post_s_scale = pyro.param(
         "post_s_scale", # dim: 1 x 1 x 1
         lambda: 1 * torch.ones(1,1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   pyro.sample(
         name = "s",
         # dim: 1 x 1 x 1 | .
         fn = dist.LogNormal(
            post_s_loc,  # dim: 1 x 1 x 1
            post_s_scale # dim: 1 x 1 x 1
         )
   )

   with pyro.plate("G", G):

      # Posterior distribution of 'base'.
      post_base_loc = pyro.param(
            "post_base_loc",
            lambda: 1 * torch.ones(1,1,G).to(device, dtype)
      )
      post_base_scale = pyro.param(
            "post_base_scale",
            lambda: 3 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      base = pyro.sample(
            name = "base",
            # dim: 1 x 1 x G | .
            fn = dist.Normal(
               post_base_loc,  # dim: 1 x 1 x G
               post_base_scale # dim: 1 x 1 x G
            )
      )

      # Posterior distribution of 'base_dev_over_N'.
      post_base_dev_over_N_loc = pyro.param(
            "post_base_dev_over_N_loc",
            lambda: -2 * torch.ones(1,1,G).to(device, dtype)
      )
      post_base_dev_over_N_scale = pyro.param(
            "post_base_dev_over_N_scale",
            lambda: .5 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      base_dev_over_N = pyro.sample(
            name = "base_dev_over_N",
            # dim: 1 x 1 x G | .
            fn = dist.LogNormal(
               post_base_dev_over_N_loc,  # dim: 1 x 1 x G
               post_base_dev_over_N_scale # dim: 1 x 1 x G
            )
      )

      # Posterior distribution of 'base_dev_over_B'.
      post_base_dev_over_B_loc = pyro.param(
            "post_base_dev_over_B_loc",
            lambda: -2 * torch.ones(1,1,G).to(device, dtype)
      )
      post_base_dev_over_B_scale = pyro.param(
            "post_base_dev_over_B_scale",
            lambda: .5 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      base_dev_over_B = pyro.sample(
            name = "base_dev_over_B",
            # dim: 1 x 1 x G | .
            fn = dist.LogNormal(
               post_base_dev_over_B_loc,  # dim: 1 x 1 x G
               post_base_dev_over_B_scale # dim: 1 x 1 x G
            )
      )

      # Posterior distribution of 'sig_dev_over_K'.
      post_sig_dev_over_K_loc = pyro.param(
            "post_sig_dev_over_K_loc",
            lambda: -2 * torch.ones(1,1,G).to(device, dtype)
      )
      post_sig_dev_over_K_scale = pyro.param(
            "post_sig_dev_over_K_scale",
            lambda: .5 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      sig_dev_over_K = pyro.sample(
            name = "sig_dev_over_K",
            # dim: 1 x 1 x G | .
            fn = dist.LogNormal(
               post_sig_dev_over_K_loc,  # dim: 1 x 1 x G
               post_sig_dev_over_K_scale # dim: 1 x 1 x G
            )
      )

      # Posterior distribution of 'sig_dev_over_N'.
      post_sig_dev_over_N_loc = pyro.param(
            "post_sig_dev_over_N_loc",
            lambda: -2 * torch.ones(1,1,G).to(device, dtype)
      )
      post_sig_dev_over_N_scale = pyro.param(
            "post_sig_dev_over_N_scale",
            lambda: .5 * torch.ones(1,1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      sig_dev_over_N = pyro.sample(
            name = "sig_dev_over_N",
            # dim: 1 x 1 x G | .
            fn = dist.LogNormal(
               post_sig_dev_over_N_loc,  # dim: 1 x 1 x G
               post_sig_dev_over_N_scale # dim: 1 x 1 x G
            )
      )

      with pyro.plate("KxG", K):
         
         # Posterior distribution of 'sig_K'.
         post_sig_K_loc = pyro.param(
               "post_sig_K_loc",
               lambda: 0 * torch.zeros(1,K,G).to(device, dtype)
         )
         sig_K = pyro.sample(
               name = "sig_K",
               # dim(sig_K): 1 x K x G | .
               fn = dist.Normal(
                  post_sig_K_loc, # dim: 1 x K x G
                  sig_dev_over_K  # dim: 1 x 1 x G
               )
         )

         with pyro.plate("NxKxG", N):

            # Posterior distribution of 'sig_KN'.
            post_sig_KN_loc = pyro.param(
                  "post_sig_KN_loc",
                  lambda: 0 * torch.zeros(N,K,G).to(device, dtype)
            )
            pyro.sample(
                  name = "sig_KN",
                  # dim(sig_KN): N x K x G | .
                  fn = dist.Normal(
                     sig_K + post_sig_KN_loc, # dim: 1 x K x G
                     sig_dev_over_N           # dim: 1 x 1 x G
                  )
            )

      with pyro.plate("NxG", N):

         # Posterior distribution of 'base_N'.
         post_base_N_loc = pyro.param(
                "post_base_N_loc",
                lambda: 0 * torch.zeros(1,N,G).to(device, dtype)
         )
         pyro.sample(
                name = "base_N",
                # dim: 1 x N x G | .
                fn = dist.Normal(
                   base + post_base_N_loc, # dim: 1 x N x G
                   base_dev_over_N         # dim: 1 x 1 x G
                )
         )

      with pyro.plate("BxG", B):

         # Posterior distribution of 'base_B'.
         post_base_B_loc = pyro.param(
                "post_base_B_loc",
                lambda: 0 * torch.zeros(1,B,G).to(device, dtype)
         )
         pyro.sample(
                name = "base_B",
                # dim: 1 x 1 x G | .
                fn = dist.Normal(
                   post_base_B_loc, # dim: 1 x B x G
                   base_dev_over_B  # dim: 1 x 1 x G
                )
         )

   with pyro.plate("ncells", ncells):

      # Posterior distribution of 'theta'.
      post_theta_loc = pyro.param(
            "post_theta_loc",
            lambda: 0 * torch.zeros(1,K,ncells).to(device, dtype),
      )
      post_theta_scale = pyro.param(
            "post_theta_scale",
            lambda: 1 * torch.ones(1,K,ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "theta",
            # dim: 1 x K x ncells | .
             fn = dist.Normal(
                post_theta_loc,  # dim: 1 x K x ncells
                post_theta_scale # dim: 1 x K x ncells
             )
      )

      # Posterior distribution of 'shift'.
      post_c_loc = pyro.param(
            "post_c_loc",
            lambda: 0 * torch.zeros(1,1,ncells).to(device, dtype),
      )
      post_c_scale = pyro.param(
            "post_c_scale",
            lambda: 1 * torch.ones(1,1,ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "shift",
            # dim: 1 x 1 x ncells
            fn = dist.Normal(
               post_c_loc,  # dim: 1 x 1 x ncells
               post_c_scale # dim: 1 x 1 x ncells
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

   pyro.set_rng_seed(123)
   torch.manual_seed(123)
   torch.use_deterministic_algorithms(True)

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
   # Setting 'max_plate_nesting' to 3 allows Pyro to add the particle
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
      "post_pi_0", "post_pi_1",
      "post_s_loc", "post_s_scale",
      "post_base_loc",
      "post_base_scale",
      "post_base_N_loc",
      "post_base_B_loc",
      "post_base_dev_over_N_loc",
      "post_base_dev_over_N_scale",
      "post_base_dev_over_B_loc",
      "post_base_dev_over_B_scale",
      "post_sig_dev_over_K_loc",
      "post_sig_dev_over_K_scale",
      "post_sig_K_loc",
      "post_sig_dev_over_N_scale",
      "post_sig_dev_over_K_loc",
      "post_sig_KN_loc",
      "post_base_N_loc",
      "post_base_B_loc",
      "post_theta_loc", "post_theta_scale",
      "post_c_loc", "post_c_scale",
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
