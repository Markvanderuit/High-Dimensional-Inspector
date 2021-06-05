import conf
from pathlib import Path
from numpy import linspace

# Paths to executables
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Varying parameter is n
n = range(1000, 61000, 1000)

# Set up MNIST config for our method
baseSneConf = conf.sneConfig()
baseSneConf.doKLD = True
baseSneConf.doVis = False
baseSneConf.doLbl = True
baseSneConf.doLog = True
baseSneConf.doNNP = True
baseSneConf.datasets = ["fashion_labeled_60k_784d"]
baseSneConf.n = [60000]
baseSneConf.hd = [784]
baseSneConf.ld = [2]
baseSneConf.iters = [1000]
baseSneConf.perp = [50]

# Set up MNIST config for CU-sne method
baseCuConf = conf.cuConfig()
baseCuConf.doKLD = True
baseCuConf.doLbl = True
baseCuConf.doLog = True
baseCuConf.doNNP = True
baseCuConf.datasets = ["fashion_labeled_60k_784d"]
baseCuConf.n = [60000]
baseCuConf.hd = [784]
baseCuConf.iters = [1000]
baseCuConf.perp = [50]

# 2D, linear
sne = baseSneConf
sne.prefix = "eval_tests/2d_linear/"
sne.theta = [0.0]
sne.theta2 = [0.0]
sne.scaling = [2.0]
sne.run(snePath)
print(sne.prefix)

# 2D, ours
sne = baseSneConf
sne.prefix = "eval_tests/2d_ours/"
sne.theta = [0.5]
sne.theta2 = [0.35]
sne.scaling = [2.0]
sne.run(snePath)
print(sne.prefix)

# 2D, cuda
sne = baseCuConf
sne.prefix = "eval_tests/2d_cuda/"
sne.run(cuPath, evalPath)

# 3D, linear
sne = baseSneConf
sne.prefix = "eval_tests/3d_linear/"
sne.ld = [3]
sne.theta = [0.0]
sne.theta2 = [0.0]
sne.scaling = [1.2]
sne.run(snePath)

# 3D, ours
sne = baseSneConf
sne.prefix = "eval_tests/3d_ours/"
sne.ld = [3]
sne.theta = [0.5]
sne.theta2 = [0.35]
sne.scaling = [1.2]
sne.run(snePath)