import package as conf
from pathlib import Path

# Path to tsne executable
snePath = Path("../../build/applications/release/tsne_cmd.exe")
cuPath = Path("../../../CUDA/build/applications/release/tsne_cmd.exe")
evalPath = Path("../../build/applications/release/evaluation_cmd.exe")

# Configure test run
sne = conf.sneConfig()
sne.doKLD = True
sne.doLbl = True
sne.datasets = ["mnist_labeled_60k_784d"]
sne.n = [40000, 50000, 60000]
sne.hd = [784]
sne.ld = [2]
sne.iters = [1000]
sne.perp = [50]
sne.theta = [0.5]
sne.theta2 = [0.0]
sne.xKey = "n"
sne.yKey = "minTime"

# Fire away
sne.run(snePath)
sne.result()

# # Configure test run
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