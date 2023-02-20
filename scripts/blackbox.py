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

   # Zero-inflation factor, 'pi'. The median is set
   # at 0.15, with 5% that 'pi' is less than 0.01 and
   # 5% that 'pi' is more than 50%.
   pi = pyro.sample(
         # dim: 1 | .
         name = "pi",
         fn = dist.Beta(
            1. * torch.ones(1).to(device, dtype),
            4. * torch.ones(1).to(device, dtype)
         )
   )

   # Variance-to-mean model. Variance is modelled as
   # 's * u^ss', where 'u' is mean gene expression and
   # 's' and 'ss' are trained parameters. 's' has 90%
   # chance of being in the interval (1.5,15) and 'ss'
   # has 90% chance of being in the interval (0.4,2.3).
   ss = pyro.sample(
      name = "ss",
      # dim: 1 | .
      fn = dist.LogNormal(
         .0 * torch.zeros(1).to(device, dtype),
         .5 * torch.ones(1).to(device, dtype)
      )
   )

   s = 1. + pyro.sample(
      name = "s",
      # dim: 1 | .
      fn = dist.LogNormal(
         1.0 * torch.ones(1).to(device, dtype),
         0.5 * torch.ones(1).to(device, dtype)
      )
   )

   with pyro.plate("G", G):

      # Batch effects (mixed) on every gene. The 'loc'
      # hyper parameter has 90% chance of being in the
      # interval (-5,5), i.e., from 0 to 150 reads.
      # The 'scale' hyper parameter has 90% chance of
      # being below 0.08, meaning that batch effects
      # are expected to be up to 25% for 99% of genes.
      hyper_b_loc = pyro.sample(
         name = "hyper_b_loc",
         # dim: 1 x G | .
         fn = dist.Normal(
            1 * torch.ones(1,G).to(device, dtype),
            1 * torch.ones(1,G).to(device, dtype)
         )
      )
      hyper_b_scale = pyro.sample(
         name = "hyper_b_scale",
         # dim: 1 x G | .
         fn = dist.HalfNormal(
            0.05 * torch.ones(1,G).to(device, dtype)
         )
      )

      # Cell type effects (mixed) on every gene. The
      # 'scale' hyper parameter has 90% chance of
      # being below 0.16, meaning that differences
      # between cell types are expected to be up to
      # 50% for 99% of genes.
      hyper_t_scale = pyro.sample(
         name = "hyper_t_scale",
         # dim: 1 x G | .
         fn = dist.HalfNormal(
            0.1 * torch.ones(1,G).to(device, dtype)
         )
      )

      # Gene modules (mixed). The 'scale' hyper
      # parameter has 90% chance of being below 0.823,
      # meaning that differences between modules are
      # expected to be up to 25% for 99% of genes.
      hyper_g_scale = pyro.sample(
         name = "hyper_g_scale",
         # dim: 1 x G | .
         fn = dist.HalfNormal(
            0.05 * torch.ones(1,G).to(device, dtype)
         )
      )

      with pyro.plate("BxG", B):
         # Batch effect (mixed). See hyper parameters.
         bmat = pyro.sample(
            name = "bmat",
            # dim: B x G | .
            fn = dist.Normal(
               hyper_b_loc,
               hyper_b_scale
            )
         )
         # Assign baseline based on batch (multiply
         # 'bmat' with a one-hot encoding of the
         # batches).
         # dim: ncells x B
         one_hot = F.one_hot(batch.to(torch.int64)).to(device, dtype)
         # dim: ncells x G
         baseline = torch.matmul(one_hot, bmat)

      with pyro.plate("NxG", N):
         # Cell type effects (mixed). See hyper parameters.
         tmat = pyro.sample(
            name = "tmat",
            # dim: N x G | .
            fn = dist.Normal(
               torch.zeros(1,G).to(device, dtype),
               hyper_t_scale
            )
         )
         # Assign cell type effects (multiply 'tmat'
         # with a one-hot encoding of the batches).
         # By definition, the average must be 0.
         # dim: ncells x N
         one_hot = F.one_hot(ctype.to(torch.int64)).to(device, dtype)
         # dim: ncells x G
         # DEBUG
         #cfx = torch.matmul(one_hot, tmat)
         cfx = 0


      with pyro.plate("KxG", K):
         # Gene modules (mixed). See hyper parameters.
         g = pyro.sample(
               name = "g",
               # dim: K x G | .
               fn = dist.Normal(
                  torch.zeros(1,1).to(device, dtype),
                  hyper_g_scale
               )
         )

   with pyro.plate("K", K):
      # Expected proportion of modules, 'alpha'. This
      # is the usual prior for topic modeling. It is
      # symmetric with average 1/K and favors strong-or-
      # completely-absent types of splits.
      alpha = pyro.sample(
            name = "alpha",
            # dim: K | .
            fn = dist.Gamma(torch.ones(K).to(device, dtype) / K, 1)
      ).unsqueeze(-2)
   
   # --------------------------------------------------------------
   # The 'unsqueeze()' statement is essential. It allows the output
   # to have one extra dimension on the "left" of the event.
   # --------------------------------------------------------------

   with pyro.plate("cells", ncells):
      # Sample module frequencies in cells. This is
      # the usual sampling for topic modeling.
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

      # Relative read count per cell, 'c'. The prior is chosen
      # so that the median is 1 with a 5% chance that 'c' is
      # less than 0.5 and a 5% chance that it is more than 5.
      c = pyro.sample(
            name = "c",
            # dim: ncells | .
            fn = dist.Normal(
               0. * torch.zeros(1).to(device, dtype),
               1. * torch.ones(1).to(device, dtype)
            )
      )

      # dim: ncells x G
      lmbd = torch.matmul(theta, g)

      # dim: ncells x G
      u = torch.exp(baseline + cfx + lmbd + c.unsqueeze(-1))

      # Variance and parameters of the negative binomial.
      # Parameter 'u' is the average number of reads and
      # the variance is 's x u^ss'. Parametrize 'r' and
      # 'p' as a function of 'u', 's' and 'ss'. Parameter
      # 'u' has dimensions ncells x G, 'ss' has dimension
      # 1 and 's' has dimension  1 x G (the result has
      # dimension ncells x G).
      p_ = 1. - 1. / (s*u**(ss-1))
      r_ = u / (s*u**(ss-1) - 1)

      # Make sure that parameters of the ZINB are valid.
      eps = 1e-6
      p = torch.clamp(p_, min=0.+eps, max=1.-eps)
      r = torch.clamp(r_, min=0.+eps)

      # Observations are sampled from a ZINB distribution.
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
         lambda: 6. * torch.ones(1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   posterior_pi = pyro.sample(
         name = "pi",
         fn = dist.Beta(
            posterior_pi_0,
            posterior_pi_1
         )
   )

   # Posterior distribution of 'ss'.
   posterior_ss_loc = pyro.param(
         "posterior_ss_loc",
         lambda: .0 * torch.ones(1).to(device, dtype)
   )
   posterior_ss_scale = pyro.param(
         "posterior_ss_scale",
         lambda: .5 * torch.ones(1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   pyro.sample(
      name = "ss",
      # dim: 1 | .
      fn = dist.LogNormal(
         posterior_ss_loc,
         posterior_ss_scale
      )
   )

   # Posterior distribution of 's'.
   posterior_s_loc = pyro.param(
         "posterior_s_loc",
         lambda: torch.ones(1).to(device, dtype)
   )
   posterior_s_scale = pyro.param(
         "posterior_s_scale",
         lambda: torch.ones(1).to(device, dtype),
         constraint = torch.distributions.constraints.positive
   )
   pyro.sample(
      name = "s",
      # dim: 1 | .
      fn = dist.LogNormal(
         posterior_s_loc,
         posterior_s_scale
      )
   )

   with pyro.plate("G", G):
      # Posterior distribution of 'hyper_b_loc'.
      posterior_hyper_b_loc_loc = pyro.param(
            "posterior_hyper_b_loc_loc",
            lambda: 3 * torch.ones(1,G).to(device, dtype)
      )
      posterior_hyper_b_loc_scale = pyro.param(
            "posterior_hyper_b_loc_scale",
            lambda: 1 * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "hyper_b_loc",
            # dim: 1 x G | .
            fn = dist.Normal(
               posterior_hyper_b_loc_loc,
               posterior_hyper_b_loc_scale
            )
      )
      # Posterior distribution of 'hyper_b_scale'.
      posterior_hyper_b_scale_loc = pyro.param(
            "posterior_hyper_b_scale_loc",
            lambda: 1 * torch.ones(1,G).to(device, dtype)
      )
      posterior_hyper_b_scale_scale = pyro.param(
            "posterior_hyper_b_scale_scale",
            lambda: 1 * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "hyper_b_scale",
            # dim: 1 x G | .
            fn = dist.LogNormal(
               posterior_hyper_b_scale_loc,
               posterior_hyper_b_scale_scale
            )
      )

      # Posterior distribution of 'hyper_g_scale'.
      posterior_hyper_g_scale_loc = pyro.param(
            "posterior_hyper_g_scale_loc",
            lambda: torch.zeros(1,G).to(device, dtype)
      )
      posterior_hyper_g_scale_scale = pyro.param(
            "posterior_hyper_g_scale_scale",
            lambda: 1 * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "hyper_g_scale",
            # dim: 1 x G | .
            fn = dist.LogNormal(
               posterior_hyper_g_scale_loc,
               posterior_hyper_g_scale_scale
            )
      )

      # Posterior distribution of 'hyper_t_scale'.
      posterior_hyper_t_scale_loc = pyro.param(
            "posterior_hyper_t_scale_loc",
            lambda: torch.zeros(1,G).to(device, dtype)
      )
      posterior_hyper_t_scale_scale = pyro.param(
            "posterior_hyper_t_scale_scale",
            lambda: 0.1 * torch.ones(1,G).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      pyro.sample(
            name = "hyper_t_scale",
            # dim: 1 x G | .
            fn = dist.LogNormal(
               posterior_hyper_t_scale_loc,
               posterior_hyper_t_scale_scale
            )
      )

      with pyro.plate("BxG", B):
         # Posterior distribution of 'b'.
         posterior_b_loc = pyro.param(
               "posterior_b_loc",
               lambda: torch.zeros(B,G).to(device, dtype)
         )
         posterior_b_scale = pyro.param(
               "posterior_b_scale",
               lambda: torch.ones(B,G).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )
         pyro.sample(
               name = "bmat",
               # dim: B x G | .
               fn = dist.Normal(
                  posterior_b_loc,
                  posterior_b_scale
               )
         )

      with pyro.plate("NxG", N):
         # Posterior distribution of 't'.
         posterior_t_loc = pyro.param(
               "posterior_t_loc",
               lambda: torch.zeros(N,G).to(device, dtype)
         )
         posterior_t_scale = pyro.param(
               "posterior_t_scale",
               lambda: torch.ones(N,G).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )
         pyro.sample(
            name = "tmat",
            # dim: N x G | .
            fn = dist.Normal(
               posterior_t_loc,
               posterior_t_scale
            )
         )
      with pyro.plate("KxG", K):
         # Posterior distribution of 'g'.
         posterior_g_loc = pyro.param(
               "posterior_g_loc",
               lambda: torch.zeros(K,G).to(device, dtype),
         )
         posterior_g_scale = pyro.param(
               "posterior_g_scale",
               lambda: torch.ones(K,G).to(device, dtype),
               constraint = torch.distributions.constraints.positive
         )
         g = pyro.sample(
               name = "g",
               # dim: K x G | .
               fn = dist.Normal(
                  posterior_g_loc,
                  posterior_g_scale
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
            lambda: 1. * torch.zeros(ncells).to(device, dtype),
      )
      posterior_c_scale = pyro.param(
            "posterior_c_scale",
            lambda: 1. * torch.ones(ncells).to(device, dtype),
            constraint = torch.distributions.constraints.positive
      )
      c = pyro.sample(
            name = "c",
            # dim: ncells
            fn = dist.Normal(
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
   for step in range(4000):
      loss += svi.step(data)
      scheduler.step()
      if (step+1) % 500 == 0:
         sys.stderr.write(f"iter {step+1}: loss = {round(loss/1e9,2)}\n")
         loss = 0.

   # Model parameters.
   names = (
      "posterior_pi_0", "posterior_pi_1",
      "posterior_s_loc", "posterior_s_scale",
      "posterior_ss_loc", "posterior_ss_scale",
      "posterior_b_loc", "posterior_b_scale",
      "posterior_c_loc", "posterior_c_scale",
      "posterior_g_loc", "posterior_g_scale",
      "posterior_t_loc", "posterior_t_scale",
      "posterior_alpha", "posterior_theta",
      "posterior_hyper_b_loc_loc", "posterior_hyper_b_loc_scale",
      "posterior_hyper_b_scale_loc", "posterior_hyper_b_scale_scale",
      "posterior_hyper_g_scale_loc", "posterior_hyper_g_scale_scale",
      "posterior_hyper_t_scale_loc", "posterior_hyper_t_scale_scale",
   )
   params = { name: pyro.param(name).detach().cpu() for name in names }
   torch.save({"params":params}, out_fname)

   # Posterior predictive sampling.
   predictive = pyro.infer.Predictive(
         model = model,
         guide = guide,
         num_samples = 1000,
         return_sites = ("theta", "g", "_RETURN"),
   )
   sim = predictive(data=(batches, ctypes, None), generate=X.shape[0])
   # Resample the full transcriptome for each cell.
   smpl = {
      "tx": sim["_RETURN"].detach().cpu(),
      "theta": sim["theta"].detach().cpu(),
   }

   # Save model and posterior predictive samples.
   torch.save({"params":params, "smpl":smpl}, out_fname)
