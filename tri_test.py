import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
from scipy.stats import multivariate_normal as mvg

ax = m3d.Axes3D(plt.figure())

vtx = np.array(((0.0,0.0,0.0),(0.0,0.0,1.0),(0.0,2.0,0.5)))
vtx = np.random.randn(3,3)
tri = m3d.art3d.Poly3DCollection([vtx])
tri.set_facecolor((0.7,0.7,1.0,0.5))
tri.set_edgecolor('k')
tri.set_alpha(0.1)

ax.add_collection3d(tri)

x = np.linspace(0,1,15)
y = np.linspace(0,1,15)

A = vtx[0,:]
B = vtx[1,:]
C = vtx[2,:]
pts = []
for i in range(len(x)):
    for j in range(i,len(y)):
        u = x[i]
        v = 1-y[j]
        #pt = (1-u) * A + u * ((1-v)*B + v*C)
        pt = A + u*(B-A) + v*(C-A)
        pts.append(pt)
pts = np.array(pts)
ax.scatter(pts[:,0],pts[:,1],pts[:,2])

cov = np.vstack(((B-A),(C-A))).T
js = np.sqrt(np.linalg.det(cov.T.dot(cov)))
print(cov,cov.shape)
print(js)


at = np.linalg.norm(np.cross((B-A),(C-A)))/2
print('TRUTH:\t',at)
print('JAC/2:\t',js/2)

u = np.random.randn(3,3)[0,:]/3
covar = np.random.randn(3,3)/2
covar = np.abs(covar.T.dot(covar))

s = np.random.multivariate_normal(u,covar,500)
ax.scatter(s[:,0],s[:,1],s[:,2])
l1 = js/2*mvg.pdf((A+B+C)/3.0,u,covar)
l2 = js/2*mvg.pdf((A+B+C)/3.0,u,covar)

print(l1)
plt.show()
