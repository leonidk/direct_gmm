import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
import pymesh

def compute_gmm(x,w=None,k=2,iter_max=10000,e_tol=1e-6):
    if w is None:
        w = np.ones(x.shape[0])
    mu = x[np.random.choice(x.shape[0],k,replace=False),:]
    sigma = np.array([(x.std()/k)*np.identity(x.shape[1]) for _ in range(k)])
    pi = np.ones(shape=k)/k

    mu_prev = mu.copy()

    for _ in range(iter_max):

        # e-step
        gamma = np.zeros(shape=(x.shape[0],k))
        for i in range(k):
            gamma_i = w * pi[i]*mvn_pdf.pdf(x,mean=mu[i],cov=sigma[i])
            gamma[:,i] = gamma_i
        gamma = gamma/gamma.sum(1,keepdims=True)

        # m-step
        for i in range(k):
            new_mu = np.zeros(x.shape[1])
            for j in range(x.shape[0]):
                new_mu += gamma[j,i] * x[j,:]
            new_mu /= gamma.sum(0)[0]
            mu[i,:] = new_mu
            new_sigma = np.identity(x.shape[1])*e_tol
            for j in range(x.shape[0]):
                xv = x[j,:][:,np.newaxis]
                xm = new_mu[:,np.newaxis]
                xd = xv - xm
                new_sigma += gamma[j,i] * (xd @ xd.T)
            new_sigma /= gamma.sum(0)[0]
            sigma[i,:,:] = new_sigma
        pi = gamma.mean(0)
        if ((mu-mu_prev)**2).sum() < 1e-6:
            break
        mu_prev = mu.copy()
    return mu,sigma,pi

mesh1 = pymesh.load_mesh("bunny/bun_zipper_res4.ply")
mesh2 = pymesh.load_mesh("bunny/bun_zipper_res4_spr.ply")
mesh3 = pymesh.load_mesh("bunny/bun_zipper_res4_pds.ply")
mesh4 = pymesh.load_mesh("bunny/bun_zipper_res4_25k_pds.ply")

def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1))
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    return centroids, areas

com,a = get_centroids(mesh1)
verts = mesh1.vertices
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
ax = m3d.Axes3D(plt.figure())
ax.scatter(com[:,0],com[:,1],com[:,2],s=(1e5*a)**2)
ax.scatter(verts[:,0],verts[:,1],verts[:,2],s=20)
plt.show()
