import sys
import matplotlib.pyplot as plt
import numpy as np

filename = sys.argv[1]
t = np.load(f'{filename}.npy').reshape((-1,))
print(np.min(t), np.max(t))

plt.hist(t, bins=np.linspace(np.min(t), np.max(t), 20))
ax = plt.gca()
ax.set_xlabel('Neuron value')
ax.set_ylabel('Frequency')
plt.savefig(f'{filename}.png')
