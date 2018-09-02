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

gm3 = GaussianMixture(100,init_params='kmeans'); gm3.set_triangles(face_vert); gm3.fit(com); gm3.set_triangles(None)

def tri_loss(gmm,faces_and_verts):
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    areas = areas/areas.sum()
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
            t1 = dev.dot(dev.T)
            t2 = (a.dot(a.T) + b.dot(b.T) + c.dot(c.T) - 3*m.dot(m.T))
            res -= 0.5 * np.trace(( t1 + (1/12.0) * t2).dot(si))
            thing[i] = res*areas[idx] + np.log(pi)
        total += logsumexp(thing)
    return total#/face.shape[0]
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

print(tri_loss(gm3,face_vert))
print(" ",pt_loss(gm3,com))
print(" ",gm3._estimate_weighted_log_prob(com).sum())
print(" ",gm3._estimate_weighted_log_prob(com).shape)

print(gm3.score(com))
