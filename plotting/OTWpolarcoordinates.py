import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

theta = np.zeros(shape=16000, dtype='float')
r = np.zeros(shape=16000, dtype='float')
dtheta = np.zeros(shape=16000, dtype='float')
ddtheta = np.zeros(shape=16000, dtype='float')
dr = np.zeros(shape=16000, dtype='float')
ddr = np.zeros(shape=16000, dtype='float')

#inicjalizacja warunków początkowych
x0 = 150.0
y0 = 200.0
vx0 = 2.0
vy0 = 15.0
theta[0] = np.arctan(y0 / x0)
r[0] = np.sqrt(x0**2 + y0**2)
dr[0] = vx0*x0/np.sqrt(x0**2 + y0**2) + vy0*y0/np.sqrt(x0**2 + y0**2)
dtheta[0] = x0*vy0/(x0**2 + y0**2) - y0*vx0/(x0**2 + y0**2)

#obliczenie początkowych przyspieszeń
ddtheta[0] = -2*dr[0]*dtheta[0]/r[0]
ddr[0] = r[0]*dtheta[0]**2

i = 1
krok = 0.004  #krok obliczeń
#obliczenia
while i < 16000:
    theta[i] = theta[i - 1] + krok * dtheta[i - 1]
    r[i] = r[i - 1] + krok * dr[i - 1]
    dtheta[i] = dtheta[i - 1] + krok * ddtheta[i - 1]
    dr[i] = dr[i - 1] + krok * ddr[i - 1]
    ddtheta[i] = -2*dr[i]*dtheta[i]/r[i]
    ddr[i] = r[i]*dtheta[i]**2
    i += 1

X = r*np.cos(theta)
Y = r*np.sin(theta)
u, v = np.mgrid[0:2*np.pi:40j, 0:500:20j]
u, v = np.mgrid[0:2*np.pi:40j, 0:np.max(r):20j]
x = v*np.cos(u)
y = v*np.sin(u)
z = np.cos(u) * 0
ax.plot_wireframe(x, y, z, color="r", linewidth=0.5)


ax.plot(X, Y, np.zeros(shape=16000), label='parametric curve', zdir='z')
fig.savefig('geodesic.png', dpi=72)
plt.show()

