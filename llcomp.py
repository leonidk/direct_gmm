import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
from cluster import MiniBatchKMeans
from mixture import GaussianMixture
import pymesh
from scipy.special import logsumexp

mesh0 = pymesh.load_mesh("bunny/bun_zipper_res4.ply")
#mesh3 = pymesh.load_mesh("bunny/bun_zipper_res4_pds.ply")
#mesh4 = pymesh.load_mesh("bunny/bun_zipper_res4_25k_pds.ply")
mesh4 = pymesh.load_mesh("bunny/bun_zipper_res4_sds.ply")

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
gm3 = GaussianMixture(20,init_params='random',tol=1e-2,max_iter=5); gm3.fit(mesh4.vertices)

def pt_loss(gmm,points):
    total = 0.0
    thing = np.zeros((points.shape[0],gmm.weights_.shape[0]))
    i = 0
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        thing[:,i] = pi*mvn_pdf.pdf(points,mu,s)
        i+=1
    return np.log(thing.sum(1)).sum()#logsumexp(thing,axis=1).mean()#/points.shape[0]
def pt_loss_lb(gmm,points):
    total = 0.0
    #for p in points:
    thing = np.zeros((points.shape[0],gmm.weights_.shape[0]))
    i = 0
    #things = 
    weights = np.zeros(thing.shape)
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        weights[:,i] = mvn_pdf.pdf(points,mu,s)
        i+=1
    row_sums = weights.sum(axis=1)
    #print(row_sums.shape)
    weights = weights / row_sums[:, np.newaxis]
    i=0
    print(weights.shape)
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        res = 0.0
        dev = points-mu

        res = 0.0
        res -= 0.5 * np.log(2*np.pi) *3
        res -= 0.5 * np.log(np.linalg.det(s))
        t1 = (dev.dot(si) * dev).sum(1)
        res -= 0.5 * t1
        #total += (res + np.log(pi)).sum()
        thing[:,i] = (res + np.log(pi))
        i+=1
    #total += thing.sum()#logsumexp(thing)
    #thing = thing*weights
    return ((thing-np.log(weights))*weights).sum()#logsumexp(thing,axis=1).mean()#/points.shape[0]


for pn in np.logspace(1,np.log10(mesh4.vertices.shape[0]*.95),10):
    scores = []
    scores2 = []
    for itern in range(2):
        ptsn = np.random.choice(range(mesh4.vertices.shape[0]),int(pn),replace=False)
        scores.append(pt_loss(gm3,mesh4.vertices[ptsn,:]))
        scores2.append(pt_loss_lb(gm3,mesh4.vertices[ptsn,:]))
        #scores.append(gm3._estimate_weighted_log_prob(mesh4.vertices[ptsn,:]).sum()/pn)
    scores = np.array(scores)
    scores2 = np.array(scores2)

    print(ptsn.shape[0],'\t',scores.mean(),'\t',scores2.mean())
#print(" ",gm3.score(mesh4.vertices))

#print(" ",gm3._estimate_weighted_log_prob(mesh4.vertices).sum()/mesh4.vertices.shape[0])
