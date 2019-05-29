import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
from mixture import GaussianMixture
import pymesh

mesh0 = pymesh.load_mesh("bunny/bun_zipper_res4.ply")
#mesh3 = pymesh.load_mesh("bunny/bun_zipper_res4_pds.ply")
mesh4 = pymesh.load_mesh("bunny/bun_zipper_res4_25k_pds.ply")

def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1))
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    return centroids, areas

def get_tri_covar(tris):
    covars = []
    for face in tris:
        A = face[0][:,None]
        B = face[1][:,None]
        C = face[2][:,None]
        M = (A+B+C)/3
        covars.append(A @ A.T + B @ B.T + C @ C.T - 3* M @ M.T)
    return np.array(covars)*(1/12.0)
com,a = get_centroids(mesh0)
face_vert = mesh0.vertices[mesh0.faces.reshape(-1),:].reshape((mesh0.faces.shape[0],3,-1))
data_covar = get_tri_covar(face_vert)

with open('bunny_1k_com_verts_tuesday_25.log','w') as fout:
    for km in [6,12,25,50,100,200,400]:
        for init in ['kmeans','random']:
            for exp_n in range(10):
                gm3 = GaussianMixture(km,init_params=init,max_iter=25,tol=1e-12); gm3.set_covars(data_covar); gm3.set_areas(a); gm3.fit(com); gm3.set_covars(None);  gm3.set_areas(None)
                gm0 = GaussianMixture(km,init_params=init,max_iter=25,tol=1e-12); gm0.set_areas(a); gm0.fit(com); gm0.set_areas(None)
                gm1 = GaussianMixture(km,init_params=init,max_iter=25,tol=1e-12); gm1.fit(com)
                gm2 = GaussianMixture(km,init_params=init,max_iter=25,tol=1e-12); gm2.fit(mesh0.vertices)
                #gm3 = GaussianMixture(km,init_params=init,max_iter=25,tol=1e-4); gm3.fit(mesh2.vertices)

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
                print('.',end='',flush=True)
                fout.write("{},{},{},{},{}\n".format(km,init,'0',s0,gm0.n_iter_))
                fout.write("{},{},{},{},{}\n".format(km,init,'1',s1,gm1.n_iter_))
                fout.write("{},{},{},{},{}\n".format(km,init,'2',s2,gm2.n_iter_))
                fout.write("{},{},{},{},{}\n".format(km,init,'3',s3,gm3.n_iter_))
print('')
#print(gm1.aic(mesh4.vertices),gm2.aic(mesh4.vertices))#,gm3.aic(mesh4.vertices))

#print((res[2] >0).sum(),(res2[2] >0).sum())
if False:
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as m3d
    ax = m3d.Axes3D(plt.figure())
    ax.scatter(com[:,0],com[:,1],com[:,2],s=a)
    ax.scatter(verts[:,0],verts[:,1],verts[:,2],s=20)
    plt.show()
