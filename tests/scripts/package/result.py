class Result:
  time = 0.0         # Measured total runtime
  simTime = 0.0      # Measured runtime of similarities
  minTime = 0.0      # Measured runtime of minimization
  kld = 0.0          # Measured KL-divergence

  def __init__(self, **args):
    self.__dict__.update(args)