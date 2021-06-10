import numpy as np
import matplotlib.pyplot as plt
mat = np.random.random((100, 100))
plt.imshow(mat, origin="lower", cmap='OrRd', interpolation='nearest')
plt.colorbar()
plt.show()