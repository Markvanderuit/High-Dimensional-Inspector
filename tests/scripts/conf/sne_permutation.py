import os
import subprocess
from pathlib import Path
import numpy as np
from .result import Result

def strToNum(str):
  try:
    return int(str)
  except ValueError:
    return float(str)

class SnePermutation:
  dataset = ""
  n = 100
  hd = 100
  ld = 2
  iters = 1000
  perp = 50
  theta = 0.5
  theta2 = 0.5
  scaling = 2.0

  def __init__(self, **args):
    self.__dict__.update(args)

  def genArgStr(self, config):
    argstr = config.prefix\
      + "lbl=" + ("true" if config.doLbl else "false")\
      + ", n=" + str(self.n) + ", hd=" + str(self.hd)\
      + ", ld=" + str(self.ld) + ", i=" + str(self.iters)\
      + ", p=" + str(self.perp) + ", t1=" + str(self.theta)\
      + ", t2=" + str(self.theta2) + ", s=" + str(self.scaling)
    return argstr

  def run(self, config, snePath):
    dataPath = Path("../data") / self.dataset / "data.bin"
    resultPath = Path("../results") / self.dataset / self.genArgStr(config)

    # Ensure results directory exists
    if not os.path.exists(resultPath):
      os.makedirs(resultPath)

    # Open file for passing program stdout
    f = open(str(resultPath / "log.txt"), 'w')

    args = [
      str(snePath),
      str(dataPath),
      str(resultPath / "embedding.bin"),
      str(self.n),
      str(self.hd),
      "-d", str(self.ld),
      "-i", str(self.iters),
      "-p", str(self.perp),
      "-t", str(self.theta),
      "-a", str(self.theta2),
      "-s", str(self.scaling),
      "--vis" if config.doVis else "",
      "--lbl" if config.doLbl else "",
      "--kld" if config.doKLD else "",
      "--nnp" if config.doNNP else "",
      "--txt", str(resultPath / "values.txt") 
    ]
    subprocess.run(args, stdout=f) if config.doLog else subprocess.run(args)

  def result(self, config):
    resultPath = Path("../results") / self.dataset / self.genArgStr(config)
    
    # Open file
    f = open(resultPath / "values.txt", "r")
    
    # Parse stored values
    results = { l.split(" ")[0] : strToNum(l.rstrip("\n").split(" ")[1]) for l in f.readlines() }
    return Result(time = (results["time"] if "time" in results else 0.0),
                  simTime = (results["simTime"] if "simTime" in results else 0.0),
                  minTime = (results["minTime"] if "minTime" in results else 0.0),
                  kld = (results["kld"] if "kld" in results else 0.0))

