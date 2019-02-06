import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from cluster import MiniBatchKMeans
from mixture import GaussianMixture
from scipy.spatial.distance import cdist,pdist
import pymesh
import pickle
from scipy.special import logsumexp
import scipy.optimize as opt
import transforms3d

SAMPLE_NUM = 100
method = 'Powell'
K = 20
SAMPLE_PTS = 453
mesh0 = pymesh.load_mesh("bunny/bun_zipper_res4.ply")
mesh_pts = pymesh.load_mesh("bunny/bun_zipper_res4_sds.ply")

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
gm_std = GaussianMixture(K,init_params='random',tol=1e-4,max_iter=100); gm_std.fit(mesh0.vertices)
gm_mesh = GaussianMixture(K,init_params='random',tol=1e-4,max_iter=100); gm_mesh.set_triangles(face_vert); gm_mesh.fit(com); gm_mesh.set_triangles(None)
full_points = mesh_pts.vertices

data_log_mesh = []
data_log_verts = []
data_log_icp = []
for n in range(SAMPLE_NUM):
    if False: # random transformations
        q = np.random.randn(4)
        q = q/np.linalg.norm(q)
        M =  transforms3d.quaternions.quat2mat(q)
        t = np.random.randn(3)*0.05
    else:
        t = np.random.rand(3)*0.1 - 0.05
        angles = np.random.rand(3)*30 - 15
        M = transforms3d.euler.euler2mat(angles[0],angles[1],angles[2])

    true_q = transforms3d.quaternions.mat2quat(M)
    indices = np.random.randint(0,full_points.shape[0],SAMPLE_PTS)
    samples= full_points[indices]
    source = samples @ M  + t
    def loss_verts(x):
        qs = x[:4]
        ts = x[4:]
        qs = qs/np.linalg.norm(qs)
        Ms = transforms3d.quaternions.quat2mat(qs)
        tpts =  (source - ts) @ Ms.T
        return -gm_std.score(tpts)
    res = opt.minimize(loss_verts,np.array([1,0,0,0,0,0,0]),method=method)
    rq = res.x[:4]
    rq = rq/np.linalg.norm(rq)
    rt = res.x[4:]
    #print(method)
    #print(np.arccos(rq.dot(true_q)),np.linalg.norm(rt-t))
    data_log_verts.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )
    def loss_mesh(x):
        qs = x[:4]
        ts = x[4:]
        qs = qs/np.linalg.norm(qs)
        Ms = transforms3d.quaternions.quat2mat(qs)
        tpts =  (source - ts) @ Ms.T
        return -gm_mesh.score(tpts)
    res = opt.minimize(loss_mesh,np.array([1,0,0,0,0,0,0]),method=method)
    rq = res.x[:4]
    rq = rq/np.linalg.norm(rq)
    rt = res.x[4:]
    data_log_mesh.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )

    icp_t = np.zeros(3)
    R = np.identity(3)
    source2 = np.copy(source)
    prev_err = 100000000
    indices2 = np.random.randint(0,full_points.shape[0],SAMPLE_PTS)
    samples_for_icp = full_points[indices2]
    while True:
        it = samples_for_icp.mean(0) - source2.mean(0)
        dist = cdist(source2+it,samples_for_icp)
        sample_idx = np.argmin(dist,1)
        matched_pts = samples_for_icp[sample_idx]
        H = (source2+it).T @  matched_pts
        u,s,vt = np.linalg.svd(H)
        rotmat = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
        #print(np.diag(cdist(source2,matched_pts)).mean())
        #print(np.diag(cdist(source2+it,matched_pts)).mean())
        source2 =  (source2 +it) @ rotmat.T
        err = np.diag(cdist(source2,matched_pts)).mean()
        #print(np.diag(cdist(source2,matched_pts)).mean())
        if np.linalg.norm(err-prev_err) < 1e-6:
            break
        prev_err = err
        icp_t += it
        R = R @ rotmat
    icp_q = transforms3d.quaternions.mat2quat(R)
    icp_t = -icp_t
    data_log_icp.append( [icp_q.dot(true_q),np.linalg.norm(icp_t-t)] )

np.savetxt('verts2.csv',np.array(data_log_verts),delimiter=',')
np.savetxt('mesh2.csv',np.array(data_log_mesh),delimiter=',')
np.savetxt('icp2.csv',np.array(data_log_icp),delimiter=',')
