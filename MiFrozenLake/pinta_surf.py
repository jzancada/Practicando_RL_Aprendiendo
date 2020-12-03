#para pintar una superficie
import pickle
import numpy as np 
import math

input = open('config.pickle', 'rb')
Q = pickle.load(input)
input.close()

# for key, value in Q.items():
#     print (key)
#     (s,a)=key
#     print(s,a)
#     print (Q[s,a])
a_space = np.array([-2,-1,0,1,2])

# matriz V
# s = t * N_Q + q
N_T = 24
N_Q = 20
N_A = 5

def s_to_obs(s):
    # s = t * N_Q + q
    q = s % N_Q
    t = round((s - q) / N_Q)
    print(s, [t,q])
    return [t,q]

V = np.zeros((N_T, N_Q))
for q in range(N_Q):
    for t in range(N_T):
        s = t * N_Q + q
        W = [Q[s,a] for a in a_space]
        obs = s_to_obs(s)
        V[obs[0], obs[1]] = np.max(W)
        print(np.max(W), V[obs[0], obs[1]])
     
print (V)
# pintar V 


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, N_Q)
Y = np.arange(0, N_T)
X, Y = np.meshgrid(X, Y)
Z = V

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(300, V.max())
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

plt.figure()
ax = fig.gca()
# Plot contour curves
cset = ax.contour(X, Y, Z, offset = 300, cmap=cm.coolwarm)

ax.clabel(cset, fontsize=9, inline=1)

plt.show()



plt.figure()
plt.contour(X, Y, Z, offset = -300)
plt.show()

plt.figure()
plt.plot(V[:,0])