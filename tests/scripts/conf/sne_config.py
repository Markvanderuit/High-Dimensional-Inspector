from .sne_permutation import SnePermutation

class SneConfig:
  snePath = ""
  dataPath = ""
  resPath = ""

  # Output config
  doVis = False
  doNNP = False
  doKLD = False
  doLog = False
  prefix = "sne, "
  postfix = ""

  # Dataset config
  doLbl = True       # Treat dataset as if it contains labels
  datasets = [""]    # Name of datasets to run on
  n = [100]          # Nr. of vectors in dataset
  hd = [100]         # Dims. of vectors in dataset

  # Minimization config
  ld = [2]           # Dims. of vectors in embedding
  iters = [1000]     # Nr. of iterations of minimization
  perp = [50]        # Used perplexity during minimization
  theta = [0.5]      # Theta param. for single-hierarchy approx.
  theta2 = [0.5]     # Theta param. for dual-hierarchy approx.
  scaling = [2.0]    # Field scaling (2.0 for 2d, 1.2 for 3d are recommended)

  # Result config (for LaTeX/Pgfplots output)
  xKey = "n"         # Test key, should match a config parameter name
  yKey = "minTime"       # Output key, should match a output data key name

  def __init__(self, sne, data, res):
    self.snePath = sne
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
      SnePermutation(dataset=dataset,
                     n=n,
                     hd=hd,
                     ld=ld,
                     iters=iters,
                     perp=perp,
                     theta=theta,
                     theta2=theta2,
                     scaling=scaling)
      for dataset in self.datasets
      for n in self.n
      for hd in self.hd
      for ld in self.ld
      for iters in self.iters
      for perp in self.perp
      for theta in self.theta
      for theta2 in self.theta2
      for scaling in self.scaling
    ]