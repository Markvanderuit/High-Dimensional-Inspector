import conf
from pathlib import Path

# Paths to executables
dataPath = Path("C:/Surfdrive/Documents/hdi/tests/data")
resPath = Path("C:/Surfdrive/Documents/hdi/tests/results")
snePath = Path.home() / "Documents/Builds/hdi/applications/release/tsne_cmd.exe"
evalPath = Path.home() / "Documents/Builds/hdi/applications/release/evaluation_cmd.exe"
cuPath = Path("C:/Drive/Documents/Tu Delft/T-SNE/CUDA/build/applications/release/tsne_cmd.exe")

# Configure test run
baseSneConf = conf.sneConfig(snePath, dataPath, resPath)
baseSneConf.doKLD = True
baseSneConf.doVis = False
baseSneConf.doLog = False

baseCuConf = conf.cuConfig(cuPath, evalPath, dataPath, resPath)
baseCuConf.doKLD = True
baseCuConf.doLog = False

# theta = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Example MNIST60k test run with 60k points
sne = baseSneConf
sne.doLbl = True
sne.datasets = ["mnist_labeled_60k_784d"] 
sne.n = [60000]
sne.hd = [784] 
sne.ld = [2]
sne.iters = [1024]
sne.perp = [50]
sne.theta = [0.5]
sne.theta2 = [0.25]
sne.scaling = [2.0]
sne.run()

print("SNE")
sne.xKey = "n"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# Example Imagenet 1M test run with 1M points
cu = baseCuConf
cu.doLbl = True
cu.datasets = ["mnist_labeled_60k_784d"]
cu.n = [60000]
cu.hd = [784]
cu.iters = [1024]
cu.perp = [50]
cu.run()

print("CUDA")
cu.xKey = "n"
cu.yKey = "minTime"
cu.result()
cu.yKey = "kld"
cu.result()