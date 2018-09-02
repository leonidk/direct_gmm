import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
from cluster import MiniBatchKMeans
from mixture import GaussianMixture
import pymesh

def compute_gmm(x,k=2,w=None,iter_max=10000,i_tol=1e-9,e_tol=1e-3):
    km = MiniBatchKMeans(k)
    if w is None:
        w = np.ones(x.shape[0])
    km.fit(x)
    labels = km.labels_
    mu = km.cluster_centers_
    
    sigma = []
    for i in range(k):
        new_sigma = np.identity(x.shape[1])*i_tol
        pts = x[(km.labels_ == i),:]
        sigma.append(np.cov(pts,rowvar=False) + new_sigma)
    sigma = np.array(sigma)
    #mu = x[np.random.choice(x.shape[0],k,replace=False),:]
    #sigma = np.array([(x.std()/k)*np.identity(x.shape[1]) for _ in range(k)])
    pi = np.ones(shape=k)/k

    mu_prev = mu.copy()
    for iternum in range(iter_max):
        # e-step
        gamma = np.zeros(shape=(x.shape[0],k))
        g2 = np.zeros(shape=(x.shape[0],k))

        for i in range(k):
            gamma_i = pi[i]*mvn_pdf.pdf(x,mean=mu[i],cov=sigma[i])
            gamma[:,i] = gamma_i
            g2[:,i] = pi[i]*mvn_pdf.pdf(x,mean=mu[i],cov=sigma[i])
        g2 = np.copy(gamma)
        #gamma = w.reshape((-1,1)) * gamma
        gamma = gamma/gamma.sum(1,keepdims=True)
        g2 = g2/g2.sum(1,keepdims=True)
        print(np.linalg.norm(gamma-g2),gamma.shape,w.shape)
        # m-step
        for i in range(k):
            new_mu = np.zeros(x.shape[1])
            for j in range(x.shape[0]):
                new_mu += gamma[j,i] * x[j,:]
            new_mu /= gamma.sum(0)[0]
            mu[i,:] = new_mu
            new_sigma = np.identity(x.shape[1])*i_tol
            for j in range(x.shape[0]):
                xv = x[j,:][:,np.newaxis]
                xm = new_mu[:,np.newaxis]
                xd = xv - xm
                new_sigma += gamma[j,i] * (xd @ xd.T)
            new_sigma /= gamma.sum(0)[0]
            sigma[i,:,:] = new_sigma
        pi = gamma.mean(0)
        if ((mu-mu_prev)**2).sum() < e_tol:
            break
        mu_prev = mu.copy()
    print(iternum)
    return mu,sigma,pi

mesh0 = pymesh.load_mesh("bunny/bun_zipper_1000_1.ply")
mesh1 = pymesh.load_mesh("bunny/bun_zipper_992_1.ply")

mesh2 = pymesh.load_mesh("bunny/bun_zipper_pts_1000_1.ply")
mesh3 = pymesh.load_mesh("bunny/bun_zipper_pds_1000_1.ply")
#mesh3 = pymesh.load_mesh("bunny/bun_zipper_res4_pds.ply")
mesh4 = pymesh.load_mesh("bunny/bun_zipper_50k.ply")

def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1))
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    return centroids, areas,face_vert

coma,aa,fv1 = get_centroids(mesh0)
com,a,fv2 = get_centroids(mesh1)

a = a/a.min()
aa = aa/aa.min()
#verts = mesh2.vertices#[np.random.choice(mesh2.vertices.shape[0], com.shape[0], replace=False), :]
#res  = compute_gmm(com,100,a)
#res2 = compute_gmm(verts,100)
#raise
with open('bunny_fit_week2-4r4.log','w') as fout:
    for km in [6,12,25,50,100,200,400,800]:
        for init in ['random']:
            for exp_n in range(10):
                gm0 = GaussianMixture(km,init_params=init); gm0.set_triangles(fv1); gm0.fit(coma); gm0.set_triangles(None)
                gm1 = GaussianMixture(km,init_params=init); gm1.set_triangles(fv2); gm1.fit(com); gm1.set_triangles(None)
                gm2 = GaussianMixture(km,init_params=init); gm2.fit(mesh3.vertices)
                gm3 = GaussianMixture(km,init_params=init); gm3.fit(mesh2.vertices)

                #gm3 = GaussianMixture(100); gm3.fit(mesh4.vertices)
                #print(coma.shape[0],com.shape[0],mesh2.vertices.shape[0],mesh3.vertices.shape[0])
                s0 = gm0.score(mesh4.vertices)
                s1 = gm1.score(mesh4.vertices)
                s2 = gm2.score(mesh4.vertices)
                s3 = gm3.score(mesh4.vertices)

                #print(gm0.n_iter_,gm1.n_iter_)
                #print(gm2.n_iter_,gm3.n_iter_)
                #print(s0,s1)
                #print(s2,s3)
                fout.write("{},{},{},{},{}\n".format(km,init,'0',s0,gm0.n_iter_))
                fout.write("{},{},{},{},{}\n".format(km,init,'1',s1,gm1.n_iter_))
                fout.write("{},{},{},{},{}\n".format(km,init,'2',s2,gm2.n_iter_))
                fout.write("{},{},{},{},{}\n".format(km,init,'3',s3,gm3.n_iter_))

#print(gm1.aic(mesh4.vertices),gm2.aic(mesh4.vertices))#,gm3.aic(mesh4.vertices))

#print((res[2] >0).sum(),(res2[2] >0).sum())
if False:
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as m3d
    ax = m3d.Axes3D(plt.figure())
    ax.scatter(com[:,0],com[:,1],com[:,2],s=a)
    ax.scatter(verts[:,0],verts[:,1],verts[:,2],s=20)
    plt.show()
