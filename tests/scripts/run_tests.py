import conf
from pathlib import Path

# Paths to executables
dataPath = Path.home() / "Documents/Repositories/hdi/tests/data"
resPath = Path("C:/Surfdrive/Documents/hdi/tests/results")
snePath = Path.home() / "Documents/Builds/hdi/applications/release_tsne_cmd.exe"
evalPath = Path.home() / "Documents/Builds/hdi/applications/release_evaluation_cmd.exe"
cuPath = Path("C:/Drive/Documents/Tu Delft/T-SNE/CUDA/build/applications/release/tsne_cmd.exe")

# Configure test run
baseSneConf = conf.sneConfig(snePath, dataPath, resPath)
baseSneConf.doKLD = False
baseSneConf.doVis = True
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

# # Example MNIST60k test run with 60k points
# sne = baseSneConf
# sne.doLbl = True
# sne.datasets = ["fashion_labeled_60k_784d"] 
# sne.n = [60000]
# sne.hd = [784] 
# sne.ld = [2]
# sne.iters = [1024]
# sne.perp = [50]
# sne.theta = [0.5]
# sne.theta2 = [0.25]
# sne.scaling = [2.0]

# # Example Imagenet 1M test run with 1M points
# sne = baseSneConf
# sne.doLbl = True
# sne.datasets = ["imagenet_labeled_1281167_128d"]
# sne.n = [1250000]
# sne.hd = [128]
# sne.ld = [3]
# sne.iters = [2048]
# sne.perp = [10]
# sne.theta = [0.5]
# sne.theta2 = [0.4]
# sne.scaling = [1.2]

# # Example W2V300m test run with 300k points
# sne = baseSneConf
# sne.doLbl = False
# sne.datasets = ["w2v_300m_300d"]
# sne.n = [1000000]
# sne.hd = [300]
# sne.ld = [3]
# sne.iters = [4000]
# sne.perp = [5]
# sne.theta = [0.5]
# sne.theta2 = [0.4]
# sne.scaling = [1.2]

# # Gather results for pgfplots
sne.run(snePath)
# sne.xKey = "theta2"
# sne.yKey = "minTime"
# sne.result()
# sne.yKey = "kld"
# sne.result()

# # Example Imagenet 1M test run with 1M points
# cu = baseCuConf
# cu.doLbl = False
# cu.datasets = ["w2v_300m_300d"]
# cu.n = [3000000]
# cu.hd = [300]
# cu.iters = [4000]
# cu.perp = [5]

# cu.run(cuPath, evalPath)

print("SNE")
sne.xKey = "theta2"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# print("CUDA")
# cu.xKey = "n"
# cu.yKey = "minTime"
# cu.result()
# cu.yKey = "kld"
# cu.result()

# # Gather results for pgfplots
# sne.run(snePath)
# sne.xKey = "theta2"
# sne.yKey = "minTime"
# sne.result()
# sne.yKey = "kld"
# sne.result()

# Example W2V300m test run with 300k points
# sne = baseSneConf
# sne.doLbl = False
# sne.datasets = ["w2v_300m_300d"]
# sne.n = [1000000]
# sne.hd = [300]
# sne.ld = [2]
# sne.iters = [2000]
# sne.perp = [5]
# sne.theta = [0.5]
# sne.theta2 = [0.0, 0.5]

# Example run with tSNE-CUDA
# cu = conf.cuConfig()
# cu.doKLD = True
# cu.doLbl = True
# cu.datasets = ["mnist_labeled_60k_784d"]
# cu.n = [60000]
# cu.hd = [784]
# cu.iters = [1000]
# cu.perp = [50]
# cu.xKey = "n"
# cu.yKey = "minTime"

# # Fire away
# cu.run(cuPath, evalPath)
# cu.result()