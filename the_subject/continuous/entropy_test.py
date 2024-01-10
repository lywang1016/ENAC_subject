import numpy as np

p = np.array([0.01, 0.01, 0.01, 0.97])
logp = np.log(p)
print(logp)
entropy = np.sum(-p*logp)
print(entropy)