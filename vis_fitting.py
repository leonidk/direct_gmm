import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
from scipy.special import logsumexp
import mpl_toolkits.mplot3d as m3d


means = [[0,0,-2],[0,2,0]]
covars = [np.diag([.1,1,.3]),np.diag([1,.2,.1])]

np.random.seed(30)
N = 30
pts = []
for m,c in zip(means,covars):
    pts.append(np.random.multivariate_normal(m,c,N))
pts = np.vstack(pts)

labels = np.random.rand(N*len(means),len(means))
labels /= labels.sum(1,keepdims=True)

for iteration in range(15):
    # m-step
    new_means = []
    new_covars = []
    new_pis = []
    for k in range(len(means)):
        weights = labels[:,k:k+1]
        weight_norm = weights.sum()
        new_mean = (weights * pts).sum(0)/weight_norm
        new_means.append(new_mean)

        t = pts - new_mean
        new_covar = (weights/weight_norm * t).T @ t
        new_covars.append(new_covar)

        new_pis.append( weight_norm.mean() )
    new_pis = np.array(new_pis)
    new_pis /= new_pis.sum()

    # e-step 
    for k in range(len(means)):
        labels[:,k] = new_pis[k]*mvn_pdf(new_means[k],new_covars[k]).pdf(pts)
    labels /= labels.sum(1,keepdims=True)

    if (iteration % 1) == 0:
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        colors = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in ['CA3542','27646B']]
        colors = np.array(colors)/255
        colors = np.array([[1,0,0],[0,0,1]])
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=20,c=labels@ colors)
        ax.set_xlim(-3.5,3.5)
        ax.set_ylim(-3.5,3.5)
        ax.set_zlim(-3.5,3.5)

        plt.title('E-Step Result',size=24,weight='demibold')
        plt.tight_layout()

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        for k in range(len(new_means)):
            mean,covar = new_means[k],new_covars[k]
            u,s,vt = np.linalg.svd(covar)
            coefs = (.002, .002, .002)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
            # Radii corresponding to the coefficients:
            rx, ry, rz = 1.7*np.sqrt(s)#s#1/np.sqrt(coefs)
            
            R_reg = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
            
            #print(eigs)
            # Set of all spherical angles:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)

            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = rx * np.outer(np.cos(u), np.sin(v)) #+ mean[0]
            y = ry * np.outer(np.sin(u), np.sin(v)) #+ mean[1]
            z = rz * np.outer(np.ones_like(u), np.cos(v)) #+ mean[2]
            
            for i in range(len(x)):
                for j in range(len(x)):
                    x[i,j],y[i,j],z[i,j] = np.dot([x[i,j],y[i,j],z[i,j]], vt) + mean    
            # Plot:
            res = ax.plot_surface(x,y,z,  color=colors[k],shade=True,linewidth=0.0,alpha=new_pis[k])
        ax.set_xlim(-3.5,3.5)
        ax.set_ylim(-3.5,3.5)
        ax.set_zlim(-3.5,3.5)

        plt.title('M-Step Result',size=24,weight='demibold')
        plt.tight_layout()
        #plt.show()
        plt.savefig('output/{:02d}.png'.format(iteration),dpi=300)