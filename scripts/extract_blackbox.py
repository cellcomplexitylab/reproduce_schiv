import pandas as pd
import sys
import torch

pt = sys.argv[1]
out = sys.argv[2]

info = torch.load(pt)

request_type = sys.argv[3]
if request_type == "gene_sample":
   idx = int(sys.argv[4])
   obj = info["smpl"]["tx"][:,:,idx].t()
   fmt = "%.0f"
elif request_type == "param":
   obj = info["params"][sys.argv[4]]
   fmt = "%.2f"

pd.DataFrame(obj.numpy()).to_csv(
      out, # sys.argv[2]
      sep = "\t",
      header = False,
      index = False,
      float_format = fmt
)
