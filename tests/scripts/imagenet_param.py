import conf
from pathlib import Path
from numpy import linspace

# Paths to executables
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Varying parameter is theta2
# theta2 = [0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875]
# theta2 = linspace(0.2, 0.65, 36, False)
# theta2 = [0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875]
# theta2 = linspace(0.10, 0.65  , 44, False)
# theta2 = linspace(0.1, 0.65, 11, False)
# scaling = linspace(0.5, 2.5625, 33, False)   # 0.5 to 2.5 in 0.125 increments

# # Configure run
# sne= conf.sneConfig()
# sne.doKLD = True
# sne.doVis = False
# sne.doLbl = True
# sne.doLog = False
# sne.datasets = ["imagenet_labeled_1281167_128d"]
# sne.prefix = "param_tests/theta2/"
# sne.n = [1000000]
# sne.hd = [128]
# sne.ld = [2]
# sne.iters = [2000]
# sne.perp = [5]
# sne.theta = [0.5]
# sne.theta2 = theta2
# sne.scaling = [2.0]
# # sne.run(snePath)

theta2 = linspace(0.125, 0.65, 42, False)

# Configure run
sne=conf.sneConfig()
sne.doKLD = True
sne.doVis = False
sne.doLbl = True
sne.doLog = False
sne.datasets = ["imagenet_labeled_1281167_128d"]
sne.prefix = "param_tests/theta2/"
sne.n = [1000000]
sne.hd = [128]
sne.ld = [3]
sne.iters = [2000]
sne.perp = [5]
sne.theta = [0.5]
sne.theta2 = theta2
sne.scaling = [1.0]
# sne.run(snePath)

# Output results
print(sne.prefix)
sne.xKey = "theta2"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()