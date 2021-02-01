from .cu_permutation import CuPermutation

class CuConfig:
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

  def run(self, snePath, evalPath):
    for perm in self.permute():
      perm.run(self, snePath, evalPath)

  def result(self):
    print("xKey=" + self.xKey + ", yKey=" + self.yKey)
    for perm in self.permute():
      result = perm.result(self)
      print("(" + str(getattr(perm, self.xKey)) + ", " + str(getattr(result, self.yKey)) + ")", end="")

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