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
gm3 = GaussianMixture(50,init_params='kmeans',tol=1e-3,max_iter=100); gm3.fit(com)

def tri_loss(gmm,faces_and_verts):
    centroids = face_vert.mean(1)
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    #areas = areas/areas.sum()
    total = 0.0
    #for idx, face in enumerate(faces_and_verts):
    #face is 3 faces with 3d locs
    #center = face.mean(0)
    #centr2 = centroids[idx,:]
    A = faces_and_verts[:,0,:]
    B = faces_and_verts[:,1,:]
    C = faces_and_verts[:,2,:]
    #m = center.reshape((-1,1))
    #thing = np.zeros(gmm.weights_.shape)
    thing = np.zeros((faces_and_verts.shape[0],gmm.weights_.shape[0]))

    i = 0
    #things = 
    weights = np.zeros(thing.shape)
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        weights[:,i] = mvn_pdf.pdf(centroids,mu,s)
        #print(mvn_pdf.pdf(points,mu,s).shape,weights.shape)
        i+=1
    row_sums = weights.sum(axis=1)
    #print(row_sums.shape)
    weights = weights / row_sums[:, np.newaxis]
    i=0    

    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        res = 0.0
        dev = (centroids - mu)
        
        res = 0.0
        res -= 0.5 * np.log(2*np.pi) *3
        res -= 0.5 * np.log(np.linalg.det(s))
        t1 = (dev.dot(si)*dev).sum(1)
        t2 = (A.dot(si)*A + B.dot(si)*B + C.dot(si)*C - 3*centroids.dot(si)*centroids).sum(1)
        #print("T1\t",t1.sum(),t1.min(),t1.max(),t1.mean())

        #print("T2\t",t2.sum(),t2.min(),t2.max(),t2.mean())
        res -= 0.5 * (t1 + (1.0/12.0) * t2)
        total += ((res + np.log(pi))).sum()
        thing[:,i] = ((res+ np.log(pi)))
        i+=1
    #total += thing.sum()*#areas[idx]#logsumexp(thing)*areas[idx]
    return logsumexp(thing,axis=1).mean()#.sum()/areas.sum()#.mean()#/points.shape[0]
    #return total/areas.sum()#faces_and_verts.shape[0]
def tri_loss_lb(gmm,faces_and_verts):
    centroids = face_vert.mean(1)
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    #areas = areas/areas.sum()
    total = 0.0
    #for idx, face in enumerate(faces_and_verts):
    #face is 3 faces with 3d locs
    #center = face.mean(0)
    #centr2 = centroids[idx,:]
    A = faces_and_verts[:,0,:]
    B = faces_and_verts[:,1,:]
    C = faces_and_verts[:,2,:]
    #m = center.reshape((-1,1))
    #thing = np.zeros(gmm.weights_.shape)
    thing = np.zeros((faces_and_verts.shape[0],gmm.weights_.shape[0]))

    i = 0
    #things = 
    weights = np.zeros(thing.shape)
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        weights[:,i] = mvn_pdf.pdf(centroids,mu,s)
        #print(mvn_pdf.pdf(points,mu,s).shape,weights.shape)
        i+=1
    row_sums = weights.sum(axis=1)
    #print(row_sums.shape)
    weights = weights / row_sums[:, np.newaxis]
    i=0    

    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        res = 0.0
        dev = (centroids - mu)
        
        res = 0.0
        res -= 0.5 * np.log(2*np.pi) *3
        res -= 0.5 * np.log(np.linalg.det(s))
        t1 = (dev.dot(si)*dev).sum(1)
        t2 = (A.dot(si)*A + B.dot(si)*B + C.dot(si)*C - 3*centroids.dot(si)*centroids).sum(1)
        #print("T1\t",t1.sum(),t1.min(),t1.max(),t1.mean())

        #print("T2\t",t2.sum(),t2.min(),t2.max(),t2.mean())
        res -= 0.5 * (t1 + (1.0/12.0) * t2)
        total += ((res + np.log(pi))).sum()
        thing[:,i] = ((res+ np.log(pi)))*areas#/areas.mean()
        i+=1
    #total += thing.sum()*#areas[idx]#logsumexp(thing)*areas[idx]
    #thing = thing*weights

    return np.sum(thing,axis=1).sum()/areas.sum()#.sum()/areas.sum()#.mean()#/points.shape[0]
    #return total/areas.sum()#faces_and_verts.shape[0]
def pt_loss(gmm,points):
    total = 0.0
    #for p in points:
    thing = np.zeros((points.shape[0],gmm.weights_.shape[0]))
    i = 0
    #things = 
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
    return logsumexp(thing,axis=1).mean()#logsumexp(thing,axis=1).mean()#/points.shape[0]
def pt_loss_lb(gmm,points):
    total = 0.0
    #for p in points:
    thing = np.zeros((points.shape[0],gmm.weights_.shape[0]))
    i = 0
    #things = 
    weights = np.zeros(thing.shape)
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        weights[:,i] = mvn_pdf.pdf(points,mu,s)
        #print(mvn_pdf.pdf(points,mu,s).shape,weights.shape)
        i+=1
    row_sums = weights.sum(axis=1)
    #print(row_sums.shape)
    weights = weights / row_sums[:, np.newaxis]
    i=0
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
    return thing.sum(axis=1).mean()#logsumexp(thing,axis=1).mean()#/points.shape[0]
def com_loss(gmm,points,areas):
    total = 0.0
    #for p in points:
    thing = np.zeros((points.shape[0],gmm.weights_.shape[0]))
    i = 0
    #things = 
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        res = 0.0
        dev = points-mu

        res = 0.0
        res -= 0.5 * np.log(2*np.pi) *3
        res -= 0.5 * np.log(np.linalg.det(s))
        t1 = (dev.dot(si) * dev).sum(1)
        res -= 0.5 * t1
        #total += (res + np.log(pi)).sum()
        thing[:,i] = (res + np.log(pi))*(areas/areas.mean())
        i+=1
    #total += thing.sum()#logsumexp(thing)
    return logsumexp(thing,axis=1).mean()#/points.shape[0]
def com_loss_lb(gmm,points,areas):
    total = 0.0
    #for p in points:
    thing = np.zeros((points.shape[0],gmm.weights_.shape[0]))
    i = 0
    #things = 
    for mu, s, si, pi in zip(gmm.means_,gmm.covariances_,gmm.precisions_,gmm.weights_):
        res = 0.0
        dev = points-mu

        res = 0.0
        res -= 0.5 * np.log(2*np.pi) *3
        res -= 0.5 * np.log(np.linalg.det(s))
        t1 = (dev.dot(si) * dev).sum(1)
        res -= 0.5 * t1
        #total += (res + np.log(pi)).sum()
        thing[:,i] = (res + np.log(pi))*(areas/areas.mean())
        i+=1
    #total += thing.sum()#logsumexp(thing)
    return np.sum(thing,axis=1).mean()#/points.shape[0]

print("tri\t",tri_loss(gm3,face_vert))
print("mpt\t",pt_loss(gm3,com))
#print("ptLB\t",pt_loss_lb(gm3,com))

print("spt\t",gm3.score(com))

print("sp\t",gm3._estimate_weighted_log_prob(com).sum())

print('com\t',com_loss(gm3,com,a))

for pn in np.logspace(1,np.log10(mesh4.vertices.shape[0]*.95),10):
    scores = []
    for itern in range(10):
        ptsn = np.random.choice(range(mesh4.vertices.shape[0]),int(pn),replace=False)
        scores.append(pt_loss(gm3,mesh4.vertices[ptsn,:]))
        #scores.append(gm3._estimate_weighted_log_prob(mesh4.vertices[ptsn,:]).sum()/pn)
    scores = np.array(scores)
    print(ptsn.shape[0],'\t',scores.mean(),'\t',scores.std())
#print(" ",gm3.score(mesh4.vertices))

print(" ",gm3._estimate_weighted_log_prob(mesh4.vertices).sum()/mesh4.vertices.shape[0])
