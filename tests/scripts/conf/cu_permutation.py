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

class CuPermutation:
  dataset = ""
  n = 100
  hd = 100
  iters = 1000
  perp = 50

  def __init__(self, **args):
    self.__dict__.update(args)

  def genArgStr(self, config):
    argstr = config.prefix\
      + "lbl=" + ("true" if config.doLbl else "false")\
      + ", n=" + str(self.n) + ", hd=" + str(self.hd)\
      + ", i=" + str(self.iters) + ", p=" + str(self.perp)
    return argstr

  def run(self, config):
    dataPath = config.dataPath / self.dataset / "data.bin"
    resultPath = config.resPath / self.dataset / self.genArgStr(config)

    # Ensure results directory exists
    if not os.path.exists(resultPath):
      os.makedirs(resultPath)

    # Open file for passing program stdout
    f = open(str(resultPath / "log.txt"), 'w')

    args = [
      str(config.snePath),
      str(dataPath),
      str(resultPath / "embedding.bin"),
      str(self.n),
      str(self.hd),
      "-i", str(self.iters),
      "-p", str(self.perp),
      "--lbl" if config.doLbl else "",
      "--txt", str(resultPath / "values.txt")
    ]
    subprocess.run(args, stdout=f) if config.doLog else subprocess.run(args)

    if (config.doNNP or config.doKLD):
      args = [
        str(config.evalPath),
        str(dataPath),
        str(resultPath / "embedding.bin"),
        str(self.n),
        str(self.hd),
        "2",
        "-p", str(self.perp),
        "--lbl" if config.doLbl else "",
        "--txt", str(resultPath / "values.temp"),
        "--kld",    
        "--nnp" if config.doNNP else "",
        resultPath / "nnp.txt" if config.doNNP else ""
      ]
      subprocess.run(args)

      # Merge output value files
      if (config.doKLD):
        fv = open(resultPath / "values.txt", "a+")
        ft = open(resultPath / "values.temp", "r")
        fv.write(ft.read())
        fv.close()
        ft.close()

  def result(self, config):
    resultPath = config.resPath / self.dataset / self.genArgStr(config)
    
    # Open file
    f = open(resultPath / "values.txt", "r")
    
    # Parse stored values
    results = { l.split(" ")[0] : strToNum(l.rstrip("\n").split(" ")[1]) for l in f.readlines() }
    return Result(time = (results["time"] if "time" in results else 0.0),
                  simTime = (results["simTime"] if "simTime" in results else 0.0),
                  minTime = (results["minTime"] if "minTime" in results else 0.0),
                  kld = (results["kld"] if "kld" in results else 0.0))

