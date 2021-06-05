from .cu_permutation import CuPermutation

class CuConfig:
  snePath = ""
  evalPath = ""
  dataPath = ""
  resPath = ""

  # Output config
  doNNP = False
  doKLD = False
  doLog = False
  prefix = "cu, "
  postfix = ""

  # Dataset config
  doLbl = True       # Treat dataset as if it contains labels
  datasets = [""]    # Name of datasets to run on
  n = [100]          # Nr. of vectors in dataset
  hd = [100]         # Dims. of vectors in dataset

  # Minimization config
  iters = [1000]     # Nr. of iterations of minimization
  perp = [50]        # Used perplexity during minimization

  # Result config (for LaTeX/Pgfplots output)
  xKey = "n"         # Test key, should match a config parameter name
  yKey = "minTime"       # Output key, should match a output data key name

  def __init__(self, sne, eval, data, res):
    self.snePath = sne
    self.evalPath = eval
    self.dataPath = data
    self.resPath = res

  def run(self):
    for perm in self.permute():
      perm.run(self)

  def result(self):
    print("Result: xKey=" + self.xKey + ", yKey=" + self.yKey)
    print("  ", end="")
    for perm in self.permute():
      result = perm.result(self)
      print("(" + str(getattr(perm, self.xKey)) + ", " + str(getattr(result, self.yKey)) + ")", end="")
    print()

  def permute(self):
    return [
      CuPermutation(dataset=dataset,
                     n=n,
                     hd=hd,
                     iters=iters,
                     perp=perp)
      for dataset in self.datasets
      for n in self.n
      for hd in self.hd
      for iters in self.iters
      for perp in self.perp
    ]