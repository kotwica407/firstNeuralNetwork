import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


fig = plt.figure()

ax = fig.gca(projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

Z = np.sin(X) * np.cos(Y)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y) = sin(x) * cos(x)")

fig.suptitle('Wykres funkcji f(x, y) = sin(x) * cos(x)')

fig.savefig('fig4.png', dpi=72)

plt.show()