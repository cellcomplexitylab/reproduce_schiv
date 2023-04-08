import numpy as np
import pyro
import pyro.distributions as dist
import random
import sys
import torch
import torch.nn.functional as F

from local import (
      ZeroInflatedNegativeBinomial,
      warmup_scheduler,
      sc_data,
)

global K # Number of modules.
global B # Number of batches.
global N # Number of types.
global D # Number of drugs.
global G # Number of genes.

pyro.enable_validation(False)


def model(data, generate=0):

   batch, ctype, drugs, X, mask = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # dim(OH_B): ncells x B
   OH_B = F.one_hot(batch.to(torch.int64)).to(device, dtype)
   # dim(OH_N): ncells x N
   OH_N = F.one_hot(ctype.to(torch.int64)).to(device, dtype)
   # dim(OH_D): ncells x N
   OH_D = F.one_hot(drugs.to(torch.int64)).to(device, dtype)

   # Variance-to-mean model. Variance is modelled as
   # 's * u', where 'u' is mean gene expression and
   # 's' is a parameter with 90% chance of being in
   # the interval 1 + (0.3, 225).
   s = 1. + pyro.sample(
         name = "s",
         # dim: 1 x 1 | .
         fn = dist.LogNormal(
            2. * torch.ones(1,1).to(device, dtype),
            2. * torch.ones(1,1).to(device, dtype)
         )
   )

   # Zero-inflation factor, 'pi'. The median is set
   # at 0.15, with 5% that 'pi' is less than 0.01 and
   # 5% that 'pi' is more than 50%.
   pi = pyro.sample(
         # dim: 1 x 1 | .
         name = "pi",
         fn = dist.Beta(
            1. * torch.ones(1,1).to(device, dtype),
            4. * torch.ones(1,1).to(device, dtype)
         )
   )

   with pyro.plate("K", K):

      # Module weight, indicating global proportion of each
      # module in transcriptomes. This is the same prior as
      # in the standard latent Dirichlet allocation.
      mod_wgt = pyro.sample(
            name = "mod_wgt",
            # dim(mod_wgt): 1 x K
            fn = dist.Gamma(
               torch.ones(1,1).to(device, dtype) / K,
               torch.ones(1,1).to(device, dtype)
            )
      )

      # dim(mod_wgt): 1 x 1 x K
      mod_wgt = mod_wgt.unsqueeze(-2)


   with pyro.plate("ncells", ncells):

      # Proportion of the modules in given transcriptomes.
      # This is the same hierarchic model as the standard
      # latend Dirichlet allocation.
      theta = pyro.sample(
            name = "theta",
            # dim(theta): 1 x ncells | K
            fn = dist.Dirichlet(
               mod_wgt # dim: 1 x 1 x K
            )
      )

      # Correction for the total number of reads in the
      # transcriptome. The shift in log space corresponds
      # to a cell-specific scaling of all the genes of
      # the transcriptome. In linear space, the median
      # is 1 by design (average 0 in log space). In
      # linear space, the scaling factor has a 90% chance
      # of being in the window (0.2, 5).
      shift = pyro.sample(
            name = "shift",
            # dim(shift): 1 x ncells | .
            fn = dist.Normal(
               0. * torch.zeros(1,1).to(device, dtype),
               1. * torch.ones(1,1).to(device, dtype)
            )
      )

      # dim(shift): ncells x 1
      shift = shift.squeeze().unsqueeze(-1)


   # Per-gene sampling.
   with pyro.plate("G", G):

      # Dummy plate to fix rightmost shapes.
      with pyro.plate("1xG", 1):

         #  | 1.00 0.99 0.99 | 0.00 0.00 ...
         #  | 0.99 1.00 0.99 | 0.00 0.00 ...
         #  | 0.99 0.99 1.00 | 0.00 0.00 ...
         #    ----- K ------
         #    ------------ D -----------

         Q = (.99 * torch.ones(K,K)).fill_diagonal_(1.)
         cor = torch.block_diag(*([Q]*D)).to(device, dtype)
         mu = torch.zeros(1,1,K*D).to(device, dtype)

         # Modules consist of K pairs groups that scale
         # over baseline gene expression, so the median
         # is set to 1 by design. There is a 90% chance
         # that the scaling is in the window (0.6, 1.5).
         # For the paired modules, there is a 90% chance
         # that the two values are within 10% of each
         # other.
         mod = pyro.sample(
               name = "mod",
               # dim(modules): 1 x G | KD
               fn = dist.MultivariateNormal(
                  0.00 * mu,
                  0.25 * cor
               )
         )

         # dim(mod): 1 x G x D x K
         mod = mod.view(mod.shape[:-1] + (D,K))

         # Matrix 'cor_NB' has the structure below.
         #  | 1.00  0.99 |  0.97  0.97 |
         #  | 0.99  1.00 |  0.97  0.97 |
         #  ----------------------------
         #  | 0.97  0.97 |  1.00  0.99 |
         #  | 0.97  0.97 |  0.99  1.00 |
         #   ---  B  ---
         #   ------  N blocks  -------
         Q = (.02 * torch.ones(B,B)).fill_diagonal_(.03)
         cor_NB = .97 + torch.block_diag(*([Q]*N)).to(device, dtype)
         mu_NB = torch.ones(1,1,N*B).to(device, dtype)

         # Baseline expression on every gene. The base
         # has 90% chance of being in the interval
         # (-4,6), i.e., from 0 to 400 reads. Several
         # adjustments apply, including the shift to
         # correct for the total number of reads in
         # the transcriptome of the cell and the modules
         # that modify the expression of some genes.
         base = pyro.sample(
               name = "base",
               # dim(base): 1 x G | N*B
               fn = dist.MultivariateNormal(
                  1 * mu_NB, # dim: N*B 
                  3 * cor_NB # dim: N*B x N*B
               )
         )

         # dim(base): 1 x G x N x B
         base = base.view(base.shape[:-1] + (N,B))

      # dim(mod_n): ncells x G
      nprll = len(mod.shape) == 4 # (not run in parallel)
      mod_n = torch.einsum("ni,ygik,xnk->ng", OH_D, mod, theta) if nprll else \
              torch.einsum("ni,...ygik,...xnk->...ng", OH_D, mod, theta)

      # dim(base_n): ncells x G
      nprll = len(base.shape) == 4 # (not run in parallel)
      base_n = torch.einsum("ni,xgij,nj->ng", OH_N, base, OH_B) if nprll else \
               torch.einsum("ni,...xgij,nj->...ng", OH_N, base, OH_B)

      # dim(u): ncells x G
      u = torch.exp(base_n + mod_n + shift)

      # Variance and parameters of the negative binomial.
      # Parameter 'u' is the average number of reads and
      # the variance is 's x u'. Parametrize 'r' and 'p'
      # as a function of 'u' and 's'. Parameter 'u' has
      # dimensions ncells x G and 's' has dimension 1
      # (the result has dimension G x ncells).
      p_ = 1. - 1. / s # dim(p_): 1
      r_ = u / (s - 1) # dim(r_): 1 x G x ncells

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

      with pyro.plate("ncellsxG", ncells):

         # Observations are sampled from a ZINB distribution.
         Y = pyro.sample(
               name = "Y",
               # dim(X): ncells x G
               fn = ZeroInflatedNegativeBinomial(
                  total_count = r, # dim: ncells x G
                  probs = p,       # dim:          1
                  gate = pi        # dim:          1
               ),
               obs = X,
               obs_mask = mask
         )

   # Return sampled transcriptome and smooth log-estimate.
   return torch.stack((Y, base_n + mod_n))


def guide(data=None, generate=0):

   batch, ctype, drugs, X, mask = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # Posterior distribution of 's'.
   post_s_loc = pyro.param(
         "post_s_loc", # dim: 1 x 1
         lambda: 2 * torch.ones(1,1).to(device, dtype)
   )
   post_s_scale = pyro.param(
         "post_s_scale", # dim: 1 x 1
         lambda: 2 * torch.ones(1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_s = pyro.sample(
         name = "s",
         # dim: 1 x 1 x 1 | .
         fn = dist.LogNormal(
            post_s_loc,  # dim: 1 x 1
            post_s_scale # dim: 1 x 1
         )
   )

   # Posterior distribution of 'pi'.
   post_pi_0 = pyro.param(
         "post_pi_0", # dim: 1 x 1
         lambda: 1. * torch.ones(1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_pi_1 = pyro.param(
         "post_pi_1", # dim: 1 x 1
         lambda: 4. * torch.ones(1,1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   post_pi = pyro.sample(
         name = "pi",
         # dim: 1 x 1 | .
         fn = dist.Beta(
            post_pi_0, # dim: 1 x 1
            post_pi_1  # dim: 1 x 1
         )
   )

   with pyro.plate("K", K):

      post_mod_wgt = pyro.param(
            "post_mod_wgt",
            lambda: torch.ones(1,K).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )

      mod_wgt = pyro.sample(
            name = "mod_wgt",
            # dim(mod_wgt): 1 x K
            fn = dist.Gamma(
               post_mod_wgt, # dim: 1 x K
               torch.ones(1,1).to(device, dtype)
            )
      )

   with pyro.plate("ncells", ncells):

      # Posterior distribution of 'theta'.
      post_theta_loc = pyro.param(
            "post_theta_loc",
            lambda: torch.ones(1,ncells,K).to(device, dtype),
            constraint = torch.distributions.constraints.greater_than(0.5)
      )
      post_theta = pyro.sample(
            name = "theta",
            # dim(theta): 1 x ncells | K
            fn = dist.Dirichlet(
               post_theta_loc # dim: 1 x ncells x K
            )
      )

      # Posterior distribution of 'shift'.
      post_shift_loc = pyro.param(
            "post_shift_loc",
            lambda: 0 * torch.zeros(1,ncells).to(device, dtype),
      )
      post_shift_scale = pyro.param(
            "post_shift_scale",
            lambda: 1 * torch.ones(1,ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      post_shift = pyro.sample(
            name = "shift",
            # dim: 1 x ncells
            fn = dist.Normal(
               post_shift_loc,  # dim: 1 x ncells
               post_shift_scale # dim: 1 x ncells
            )
      )

   with pyro.plate("G", G):

      with pyro.plate("1xG", 1):

         # Posterior distribution of 'mod'.
         post_mod_loc = pyro.param(
               "post_mod_loc",
               lambda: 0 * torch.zeros(1,G,K*D).to(device, dtype)
         )
         post_mod_scale = pyro.param(
               "post_mod_scale",
               lambda: .25 * torch.ones(1,G,1).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )

         post_mod = pyro.sample(
               name = "mod",
               # dim: 1 x G | 2K
               fn = dist.Normal(
                  post_mod_loc,  # dim: 1 x G x 2K
                  post_mod_scale # dim: 1 x G x 2K
               ).to_event(1)
         )

         # Posterior distribution of 'base'.
         post_base_loc = pyro.param(
               "post_base_loc",
               lambda: 1 * torch.ones(1,G,N*B).to(device, dtype)
         )
         post_base_scale = pyro.param(
               "post_base_scale",
               lambda: 3 * torch.ones(1,G,1).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )

         post_base = pyro.sample(
               name = "base",
               # dim: 1 x G | N*B
               fn = dist.Normal(
                  post_base_loc,  # dim: 1 x G x N*B
                  post_base_scale # dim: 1 x G x N*B
               ).to_event(1)
         )


if __name__ == "__main__":

   pyro.set_rng_seed(123)
   torch.manual_seed(123)

   K = int(sys.argv[1])
   in_fname = sys.argv[2]
   out_fname = sys.argv[3]

   # Read in the data and set the dimensions.
   data = sc_data(in_fname)

   cells, batches, ctypes, drugs, X = data
   mask = torch.ones_like(X).to(dtype=torch.bool)
   # HSPA8 and MT-ND4 show the strongest batch
   # effects when considering SAHA and PMA
   # treatments separately.
   # Mask HSPA8 (idx 1236) and MT-ND4 (idx 4653).
   # mask[:,1236] = X[:,4653] = False
   # Mask HIV (last idx) in Jurkat cells only.
   mask[ctypes == 1,-1] = False

   B = int(batches.max() + 1)
   N = int(ctypes.max() + 1)
   D = int(drugs.max() + 1)
   G = int(X.shape[-1])

   data = batches, ctypes, drugs, X, mask

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
#      loss = pyro.infer.Trace_ELBO(
         num_particles = 16,
         vectorize_particles = True,
      )
   )

   loss = 0.
   for step in range(4000):
      loss += svi.step(data)
      scheduler.step()
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,3)}\n")
         loss = 0.

   # Model parameters.
   names = (
      "post_s_loc", "post_s_scale",
      "post_pi_0", "post_pi_1",
      "post_mod_wgt", "post_theta_loc",
      "post_base_loc", "post_base_scale",
      "post_mod_loc", "post_mod_scale",
      "post_shift_loc", "post_shift_scale",
   )
   ready = lambda x: x.detach().cpu().squeeze()
   params = { name: ready(pyro.param(name)) for name in names }

   # Posterior predictive sampling.
   predictive = pyro.infer.Predictive(
         model = model,
         guide = guide,
         num_samples = 1000,
         return_sites = ("theta", "mod", "base", "_RETURN"),
   )
   # Resample transcriptome (and smoothed version as well).
   sim = predictive(data=(batches, ctypes, drugs, None, mask), generate=X.shape[0])
   smpl = { "tx": sim["_RETURN"].detach().cpu() }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
