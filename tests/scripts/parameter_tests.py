import conf
from pathlib import Path
from numpy import linspace

# Paths to executables
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Configure test run
baseSneConf = conf.sneConfig()
baseSneConf.doKLD = True
baseSneConf.doVis = False
baseSneConf.doLbl = True
baseSneConf.doLog = True
baseSneConf.datasets = ["mnist_labeled_60k_784d"]
baseSneConf.n = [60000]
baseSneConf.hd = [784]
baseSneConf.iters = [1000]
baseSneConf.perp = [50]
baseSneConf.theta = [0.5]

# Varying parameter values we test for
theta2 = linspace(0.1, 1.05, 19, False)     # 0.1 to 1.0 in 0.05 increments
scaling = linspace(0.5, 2.625, 17, False)   # 0.5 to 2.5 in 0.125 increments

# # First test: vary theta2 from 0.1 to 1 given 2d and fieldScaling = 2.0
# sne = baseSneConf
# sne.prefix = "parameter_tests/test0/"
# sne.ld = [2]
# sne.scaling = [2.0]
# sne.theta2 = theta2
# # sne.run(snePath)

# # Gather results for pgfplots
# print(sne.prefix)
# sne.xKey = "theta2"
# sne.yKey = "minTime"
# sne.result()
# sne.yKey = "kld"
# sne.result()

# # Second test: vary theta2 from 0.1 to 1 given 3d and fieldScaling = 1.4
# sne = baseSneConf
# sne.prefix = "parameter_tests/test1/"
# sne.ld = [3]
# sne.scaling = [1.4]
# sne.theta2 = theta2
# # sne.run(snePath)

# # Gather results for pgfplots
# print(sne.prefix)
# sne.xKey = "theta2"
# sne.yKey = "minTime"
# sne.result()
# sne.yKey = "kld"
# sne.result()

# Third test: vary scaling from 0.5 to 2.5 given 2d and theta2 = 0.35
sne = baseSneConf
sne.prefix = "parameter_tests/test2/"
sne.ld = [2]
sne.scaling = scaling
sne.theta2 = [0.3]
# sne.run(snePath)

# Gather results for pgfplots
print(sne.prefix)
sne.xKey = "scaling"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# Fourth test: vary scaling from 0.5 to 2.5 given 3d and theta2 = 0.35
sne = baseSneConf
sne.prefix = "parameter_tests/test3/"
sne.ld = [3]
sne.scaling = scaling
sne.theta2 = [0.3]
# sne.run(snePath)

# Gather results for pgfplots
print(sne.prefix)
sne.xKey = "scaling"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()