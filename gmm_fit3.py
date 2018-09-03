import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
from cluster import MiniBatchKMeans
from mixture import GaussianMixture
import pymesh
from scipy.special import logsumexp

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

com,a = get_centroids(mesh0)
face_vert = mesh0.vertices[mesh0.faces.reshape(-1),:].reshape((mesh0.faces.shape[0],3,-1))

#gm3 = GaussianMixture(100,init_params='kmeans'); gm3.set_triangles(face_vert); gm3.fit(com); gm3.set_triangles(None)
gm3 = GaussianMixture(400,init_params='kmeans',tol=1e-5,max_iter=1); gm3.fit(mesh4.vertices)

def tri_loss(gmm,faces_and_verts):
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    #areas = areas/areas.sum()
    total = 0.0
    for idx, face in enumerate(faces_and_verts):
        #face is 3 faces with 3d locs
        center = face.mean(0)
        centr2 = centroids[idx,:]
        A = face[0,:]
        B = face[1,:]
        C = face[2,:]
        m = center.reshape((-1,1))
        thing = np.zeros(gmm.weights_.shape)
        i = 0
        for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
            res = 0.0
            dev = (m - mu[:,np.newaxis]).reshape((-1,1))
            a = A.reshape((-1,1))
            b = B.reshape((-1,1))
            c = C.reshape((-1,1))
            m = m.reshape((-1,1))

            res = 0.0
            res -= 0.5 * np.log(2*np.pi) *3
            res -= 0.5 * np.log(np.linalg.det(s))
            t1 = dev.T.dot(si).dot(dev)
            #t2 = (a.dot(a.T) + b.dot(b.T) + c.dot(c.T) - 3*m.dot(m.T))
            t2 = a.T.dot(si).dot(a) + b.T.dot(si).dot(b) + c.T.dot(si).dot(c) - 3*m.T.dot(si).dot(m)
            res -= 0.5 * (t1+ (1.0/12.0) * t2)
            thing[i] = (res + np.log(pi))
            i+=1
        total += logsumexp(thing)*areas[idx]
    return total/areas.sum()#faces_and_verts.shape[0]
def pt_loss(gmm,points):
    total = 0.0
    for p in points:
        thing = np.zeros(gmm.weights_.shape)
        i = 0
        for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
            res = 0.0
            dev = (p[:,np.newaxis] - mu[:,np.newaxis]).reshape((-1,1))

            res = 0.0
            res -= 0.5 * np.log(2*np.pi) *3
            res -= 0.5 * np.log(np.linalg.det(s))
            t1 = dev.T.dot(si).dot(dev)
            res -= 0.5 * t1.sum()
            thing[i] = res + np.log(pi)
            i+=1
        total += logsumexp(thing)
    return total/points.shape[0]
def com_loss(gmm,points,areas):
    total = 0.0
    for idx,p in enumerate(points):
        thing = np.zeros(gmm.weights_.shape)
        i = 0
        for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
            res = 0.0
            dev = (p[:,np.newaxis] - mu[:,np.newaxis]).reshape((-1,1))

            res = 0.0
            res -= 0.5 * np.log(2*np.pi) *3
            res -= 0.5 * np.log(np.linalg.det(s))
            t1 = dev.T.dot(si).dot(dev)
            res -= 0.5 * t1.sum()
            thing[i] = res + np.log(pi)
            i+=1
        total += logsumexp(thing)*areas[idx]
    return total/areas.sum()

print(tri_loss(gm3,face_vert))
print(" ",pt_loss(gm3,com))
print(" ",gm3.score(com))

print(" ",gm3._estimate_weighted_log_prob(com).sum())

print(com_loss(gm3,com,a))

for pn in np.linspace(5,mesh4.vertices.shape[0]/25,10):
    scores = []
    for itern in range(10):
        ptsn = np.random.choice(range(mesh4.vertices.shape[0]),int(pn),replace=False)
        scores.append(gm3.score(mesh4.vertices[ptsn,:]))
    scores = np.array(scores)
    print(ptsn.shape[0],'\t',scores.mean(),'\t',scores.std())

print(" ",gm3._estimate_weighted_log_prob(mesh4.vertices).sum()/mesh4.vertices.shape[0])
