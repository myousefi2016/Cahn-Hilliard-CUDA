import numpy as np
import matplotlib.pyplot as plt

c = np.genfromtxt('../out/integral_c.txt',delimiter=',',dtype=np.float)
mu = np.genfromtxt('../out/integral_mu.txt',delimiter=',',dtype=np.float)
f = np.genfromtxt('../out/integral_f.txt',delimiter=',',dtype=np.float)

c = c.T
mu = mu.T
f = f.T

plt.plot(c[0],c[1])
plt.xlabel('time')
plt.ylabel('c')
plt.tight_layout()
plt.show()

plt.plot(mu[0],mu[1])
plt.xlabel('time')
plt.ylabel('$\mu$')
plt.tight_layout()
plt.show()

plt.plot(f[0],f[1])
plt.xlabel('time')
plt.ylabel('F')
plt.tight_layout()
plt.show()
