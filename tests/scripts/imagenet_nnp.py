import conf
from pathlib import Path
from numpy import linspace

# Paths to executables
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Set up MNIST config for our method
baseSneConf = conf.sneConfig()
baseSneConf.doKLD = True
baseSneConf.doNNP = True
baseSneConf.doVis = False
baseSneConf.doLbl = True
baseSneConf.doLog = True
baseSneConf.datasets = ["imagenet_labeled_1281167_128d"]
baseSneConf.n = [200000]
baseSneConf.hd = [128]
baseSneConf.ld = [2]
baseSneConf.iters = [2000]
baseSneConf.perp = [20]

# Set up MNIST config for CU-sne method
baseCuConf = conf.cuConfig()
baseCuConf.doKLD = True
baseCuConf.doNNP = True
baseCuConf.doLbl = True
baseCuConf.doLog = True
baseCuConf.datasets = ["imagenet_labeled_1281167_128d"]
baseCuConf.n = [200000]
baseCuConf.hd = [128]
baseCuConf.iters = [2000]
baseCuConf.perp = [20]

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
print(sne.prefix)

# 3D, linear
sne = baseSneConf
sne.prefix = "eval_tests/3d_linear/"
sne.ld = [3]
sne.theta = [0.0]
sne.theta2 = [0.0]
sne.scaling = [1.2]
sne.run(snePath)
print(sne.prefix)

# 3D, ours
sne = baseSneConf
sne.prefix = "eval_tests/3d_ours/"
sne.ld = [3]
sne.theta = [0.5]
sne.theta2 = [0.35]
sne.scaling = [1.2]
sne.run(snePath)
print(sne.prefix)