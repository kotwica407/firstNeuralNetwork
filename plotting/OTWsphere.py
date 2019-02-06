import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

theta = np.zeros(shape=16000, dtype='float')
fi = np.zeros(shape=16000, dtype='float')
dtheta = np.zeros(shape=16000, dtype='float')
ddtheta = np.zeros(shape=16000, dtype='float')
dfi = np.zeros(shape=16000, dtype='float')
ddfi = np.zeros(shape=16000, dtype='float')

#inicjalizacja warunków początkowych
theta[0] = 0.2
fi[0] = 0.2
dtheta[0] = 0
dfi[0] = 1

#obliczenie początkowych przyspieszeń
ddtheta[0] = np.sin(theta[0]) * np.cos(theta[0]) * fi[0] * fi[0]
ddfi[0] = -2 * np.cos(theta[0]) * dtheta[0] * dfi[0] / np.sin(theta[0])

i = 1
krok = 0.0004  #krok obliczeń
#obliczenia
while i < 16000:
    theta[i] = theta[i - 1] + krok * dtheta[i - 1]
    fi[i] = fi[i - 1] + krok * dfi[i - 1]
    dtheta[i] = dtheta[i - 1] + krok * ddtheta[i - 1]
    dfi[i] = dfi[i - 1] + krok * ddfi[i - 1]
    ddtheta[i] = np.sin(theta[i]) * np.cos(theta[i]) * fi[i] * fi[i]
    ddfi[i] = -2 * np.cos(theta[i]) * dtheta[i] * dfi[i] / np.sin(theta[i])
    i += 1

X = np.cos(theta) * np.cos(fi)
Y = np.cos(theta) * np.sin(fi)
Z = np.sin(theta)

u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r", linewidth=0.5)


ax.plot(X, Y, Z, label='parametric curve')
fig.savefig('geodesic.png', dpi=72)
plt.show()

