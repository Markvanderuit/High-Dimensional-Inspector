import conf
from pathlib import Path

# Paths to executables
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Configure test run
baseSneConf = conf.sneConfig()
baseSneConf.doKLD = True
baseSneConf.doVis = False
# baseSneConf.doLog = True

# Example MNIST60k test run with 60k points
sne = baseSneConf
sne.prefix = "parameter_tests/"
sne.doLbl = True
sne.datasets = ["mnist_labeled_60k_784d"]
sne.n = [60000]
sne.hd = [784]
sne.ld = [2]
sne.iters = [1000]
sne.perp = [50]
sne.theta = [0.5]
sne.theta2 = [0.35]
sne.scaling = [1.0, 2.0]

# Example W2V300m test run with 300k points
# sne = baseSneConf
# sne.doLbl = False
# sne.datasets = ["w2v_300m_300d"]
# sne.n = [300000]
# sne.hd = [300]
# sne.ld = [2]
# sne.iters = [2000]
# sne.perp = [15]
# sne.theta = [0.5]
# sne.theta2 = [0.25]

# # Gather results for pgfplots
# sne.run(snePath)
sne.xKey = "theta2"
sne.yKey = "minTime"
sne.result()
sne.yKey = "kld"
sne.result()

# # Example Imagenet1M test run with 1M points
# sne = baseSneConf
# sne.doLbl = True
# sne.datasets = ["imagenet_labeled_1281167_128d"]
# sne.n = [1281167]
# sne.hd = [128]
# sne.ld = [2]
# sne.iters = [2000]
# sne.perp = [10]
# sne.theta = [0.5]
# sne.theta2 = [0.3]

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