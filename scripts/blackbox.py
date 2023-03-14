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

class NegativeBinomial(torch.distributions.NegativeBinomial):
   def log_prob(self, value):
      if self._validate_args:
          self._validate_sample(value)
      log_unnormalized_prob = (self.total_count * F.logsigmoid(-self.logits) +
                               value * F.logsigmoid(self.logits))
      log_normalization = (-torch.lgamma(self.total_count + value).nan_to_num(nan=float("nan"), posinf=0, neginf=-float("inf")) +
            torch.lgamma(1. + value) + torch.lgamma(self.total_count))
      return log_unnormalized_prob - log_normalization


class ZeroInflatedNegativeBinomial(dist.ZeroInflatedNegativeBinomial):
   def __init__(self, total_count, *, probs=None, logits=None, gate=None, gate_logits=None, validate_args=None):
      base_dist = NegativeBinomial(
         total_count=total_count,
         probs=probs,
         logits=logits,
         validate_args=False,
      )
      base_dist._validate_args = validate_args
      super(dist.ZeroInflatedNegativeBinomial, self).__init__(
         base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
      )


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
   OH_B = F.one_hot(batch.to(torch.int64)).to(device, dtype)
   # dim(one_hot_c): ncells x N
   OH_N = F.one_hot(ctype.to(torch.int64)).to(device, dtype)

   # Variance-to-mean model. Variance is modelled as
   # 's * u', where 'u' is mean gene expression and
   # 's' is a trained parameters with 90% chance of
   # being in the interval (1.5,15).
   s = 1. + pyro.sample(
         name = "s",
         # dim: 1 x 1 | .
         fn = dist.LogNormal(
            1.0 * torch.ones(1,1).to(device, dtype),
            0.5 * torch.ones(1,1).to(device, dtype)
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

   with pyro.plate("ncells", ncells):

      theta = pyro.sample(
            name = "theta",
            # dim(theta): K x ncells | .
            fn = dist.Normal(
               0 * torch.zeros(K,1).to(device, dtype),
               1 * torch.ones(K,1).to(device, dtype)
            )
      )

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

      with pyro.plate("1xG", 1):

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

         # Base expression on every gene. The
         # base has 90% chance of being in the interval
         # (-4,6), i.e., from 0 to 400 reads (many
         # adjustments will apply, see below).
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

         # dim(base_n): ncells x G
         nprll = len(base.shape) == 4 # (not run in parallel)
         base_n = torch.einsum("ni,xgij,nj->ng", OH_N, base, OH_B) if nprll else \
                  torch.einsum("ni,...xgij,nj->...ng", OH_N, base, OH_B)

         # Matrix 'cor_K' has the structure below, where
         # r = 1/2(K-1). The maximum anti-correlation
         # is 1/(K-1) for a matrix of size K, so we set
         # r to half of the possible maximum.
         #  | 1.  -r  -r |
         #  | -r  1.  -r |
         #  | -r  1.  1. |
         #   ---  K  ---
         cor_K = ( (-.5/(K-1) * torch.ones(K,K)).
                        fill_diagonal_(1.).to(device, dtype) )
         mu_K = torch.zeros(1,K).to(device, dtype)

         # Modules / signatures.
         mod = pyro.sample(
               name = "mod",
               # dim(modules): 1 x G | K
               fn = dist.MultivariateNormal(
                  0    * mu_K, # dim: K
                  0.25 * cor_K.unsqueeze(0) # dim: K x K
               )
         )

      # dim(mod_n): ncells x G
      nprll = len(theta.shape) == 2 # (not run in parallel)
      mod_n = torch.einsum("kn,xgk->ng", theta, mod) if nprll else \
              torch.einsum("...kn,...xgk->...ng", theta, mod)

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
         return pyro.sample(
               name = "X",
               # dim(X): ncells x G
               fn = ZeroInflatedNegativeBinomial(
                  total_count = r, # dim: ncells x G
                  probs = p,       # dim:          1
                  gate = pi        # dim:          1
               ),
               obs = X
         )


def guide(data=None, generate=0):

   batch, ctype, X = data
   device = "cuda" if X is None else X.device
   dtype = torch.float64 if X is None else X.dtype
   ncells = X.shape[0] if X is not None else generate

   # Posterior distribution of 's'.
   post_s_loc = pyro.param(
         "post_s_loc", # dim: 1 x 1
         lambda: 1 * torch.ones(1,1).to(device, dtype)
   )
   post_s_scale = pyro.param(
         "post_s_scale", # dim: 1 x 1
         lambda: 1 * torch.ones(1,1).to(device, dtype),
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

   with pyro.plate("ncells", ncells):

      # Posterior distribution of 'theta'.
      post_theta_loc = pyro.param(
            "post_theta_loc",
            lambda: 0 * torch.zeros(K,ncells).to(device, dtype),
      )
      post_theta_scale = pyro.param(
            "post_theta_scale",
            lambda: 1 * torch.ones(1,ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      post_theta = pyro.sample(
            name = "theta",
            # dim(theta): K x ncells | .
            fn = dist.Normal(
               post_theta_loc,  # dim: K x ncells
               post_theta_scale # dim: K x 1
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

         # Posterior distribution of 'base'.
         post_base_loc = pyro.param(
               "post_base_loc",
               lambda: 1 * torch.ones(1,G,N*B).to(device, dtype)
         )
         post_base_scale = pyro.param(
               "post_base_scale",
               lambda: 3 * torch.ones(1,G,N*B).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )

         post_base = pyro.sample(
               name = "base",
               # dim: 1 x G | N*B
               fn = dist.MultivariateNormal(
                  post_base_loc,               # dim: 1 x G x N*B
                  post_base_scale.diag_embed() # dim: 1 x G x N*B x N*B
               )
         )

         # Posterior distribution of 'mod'.
         post_mod_loc = pyro.param(
               "post_mod_loc",
               lambda: 0 * torch.zeros(1,G,K).to(device, dtype)
         )
         post_mod_scale = pyro.param(
               "post_mod_scale",
               lambda: .25 * torch.ones(1,G,1).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )

         post_mod = pyro.sample(
               name = "mod",
               # dim: 1 x G | K
               fn = dist.Normal(
                  post_mod_loc,  # dim: G x K
                  post_mod_scale # dim: G x 1
               ).to_event(1)
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
#      loss = pyro.infer.Trace_ELBO(
         num_particles = 4,
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
      "post_theta_loc", "post_theta_scale",
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
         return_sites = ("theta", "_RETURN"),
   )
   sim = predictive(data=(batches, ctypes, None), generate=X.shape[0])
   # Resample the full transcriptome for each cell.
   smpl = { "tx": sim["_RETURN"].detach().cpu() }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
