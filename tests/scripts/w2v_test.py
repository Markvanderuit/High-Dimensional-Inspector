import conf
from pathlib import Path
from numpy import linspace

# Paths to executables
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Varying parameter is n
n = list(range(100000, 3100000, 100000))
n.insert(0, 50000)
perp = [5]
iters = [4000]
hd = [300]

# Set up MNIST config for our method
baseSneConf = conf.sneConfig()
baseSneConf.doKLD = True
baseSneConf.doVis = False
baseSneConf.doLbl = False
baseSneConf.doLog = True
baseSneConf.datasets = ["w2v_300m_300d"]
baseSneConf.n = n
baseSneConf.hd = hd
baseSneConf.ld = [2]
baseSneConf.iters = iters
baseSneConf.perp = perp

# Set up MNIST config for CU-sne method
baseCuConf = conf.cuConfig()
baseCuConf.doKLD = True
baseCuConf.doLbl = False
baseCuConf.doLog = True
baseCuConf.datasets = ["w2v_300m_300d"]
baseCuConf.n = n
baseCuConf.hd = hd
baseCuConf.iters = iters
baseCuConf.perp = perp

# 2D, linear
sne = baseSneConf
sne.prefix = "eval_tests/2d_linear/"
sne.theta = [0.0]
sne.theta2 = [0.0]
sne.scaling = [2.0]
# sne.run(snePath)
print(sne.prefix)
sne.xKey = "n"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# 2D, ours
sne = baseSneConf
sne.prefix = "eval_tests/2d_ours/"
sne.theta = [0.5]
sne.theta2 = [0.25]
sne.scaling = [2.0]
# sne.run(snePath)
print(sne.prefix)
sne.xKey = "n"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# 2D, cuda
sne = baseCuConf
sne.prefix = "eval_tests/2d_cuda/"
# sne.run(cuPath, evalPath)
print(sne.prefix)
sne.xKey = "n"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# 3D, linear
sne = baseSneConf
sne.prefix = "eval_tests/3d_linear/"
sne.ld = [3]
sne.theta = [0.0]
sne.theta2 = [0.0]
sne.scaling = [1.2]
sne.n=[50000, 100000, 150000, 200000] #, 250000, 300000, 350000, 400000, 450000, 500000]
# sne.run(snePath)
print(sne.prefix)
sne.xKey = "n"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# 3D, ours
sne = baseSneConf
sne.prefix = "eval_tests/3d_ours/"
sne.ld = [3]
sne.theta = [0.5]
sne.theta2 = [0.4]
sne.scaling = [1.2]
sne.perp = [5]
sne.n = n
# sne.run(snePath)
print(sne.prefix)
sne.xKey = "n"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()