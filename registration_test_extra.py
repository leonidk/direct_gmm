import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mixture import GaussianMixture
from scipy.spatial.distance import cdist,pdist
import pymesh
import pickle
from scipy.special import logsumexp
import scipy.optimize as opt
import transforms3d
import time

SAMPLE_NUM = 25
method = None#'CG'#None#'CG'#None#CG'
K = 100
ICP_ITERS = 50000 #150
ICP_THRESH = 1e-9

#mesh0 = pymesh.load_mesh("bunny/bun_zipper.ply") 
#mesh_pts = pymesh.load_mesh("bunny/bun_zipper_50k.ply")
#mesh0 = pymesh.load_mesh("bunny/bun_zipper_1000_1.ply") 
#mesh_pts = pymesh.load_mesh("bunny/bun_zipper_50k.ply")
if False:    
    mesh0 = pymesh.load_mesh("bunny/bun_zipper_res4.ply")
    mesh_pts = pymesh.load_mesh("bunny/bun_zipper_res4_25k_pds.ply")
    tag = 'bunny'
elif False:
    mesh0 = pymesh.load_mesh("arma/Armadillo_1000.ply")
    mesh_pts = pymesh.load_mesh("arma/Armadillo_25k.ply")
    tag = 'arma'

elif False:
    mesh0 = pymesh.load_mesh("dragon/dragon_recon/dragon_vrip_1000.ply")
    mesh_pts = pymesh.load_mesh("dragon/dragon_recon/dragon_vrip_25k.ply")
    tag = 'dragon'

elif False:
    mesh0 = pymesh.load_mesh("happy/happy_recon/happy_vrip_1000.ply")
    mesh_pts = pymesh.load_mesh("happy/happy_recon/happy_vrip_25k.ply")
    tag = 'happy'

elif True:
    mesh0 = pymesh.load_mesh("lucy/lucy_1000.ply")
    mesh_pts = pymesh.load_mesh("lucy/lucy_25k.ply")
    tag = 'lucy'

SAMPLE_PTS = mesh0.vertices.shape[0] # number of vertecies!

def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1))
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    #face_vert = ((face_vert.shape[0]/SAMPLE_PTS)*(face_vert.reshape((-1,9))-np.repeat(centroids,3,axis=1)) + np.repeat(centroids,3,axis=1)).reshape((-1,3,3))
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
print(com.shape)
face_vert = mesh0.vertices[mesh0.faces.reshape(-1),:].reshape((mesh0.faces.shape[0],3,-1))
data_covar = get_tri_covar(face_vert)
print(data_covar.shape)

indices2 = np.random.randint(0,mesh_pts.vertices.shape[0],SAMPLE_PTS)
samples_for_icp = mesh0.vertices#np.copy(mesh_pts.vertices[indices2])
#gm3 = GaussianMixture(100,init_params='kmeans'); gm3.set_triangles(face_vert); gm3.fit(com); gm3.set_triangles(None)
#usually tol=1e-4,max_iter=100
t1 = time.time()
gm_std_km = GaussianMixture(K,init_params='kmeans',tol=1e-5,max_iter=100); gm_std_km.fit(samples_for_icp)
print((time.time()-t1)*1000)
t1 = time.time()
gm_std = GaussianMixture(K,init_params='random',tol=1e-5,max_iter=100); gm_std.fit(samples_for_icp)
print((time.time()-t1)*1000)
t1 = time.time()
#indices3 = np.random.randint(0,mesh0.vertices.shape[0],SAMPLE_PTS)
gm_mesh = GaussianMixture(K,init_params='random',tol=1e-5,max_iter=100); gm_mesh.set_covars(data_covar); gm_mesh.set_areas(a); gm_mesh.fit(com); gm_mesh.set_covars(None);  gm_mesh.set_areas(None)
print((time.time()-t1)*1000)
t1 = time.time()
#indices3 = np.random.randint(0,mesh0.vertices.shape[0],SAMPLE_PTS)
gm_mesh_kmeans = GaussianMixture(K,init_params='kmeans',tol=1e-5,max_iter=100); gm_mesh_kmeans.set_covars(data_covar); gm_mesh_kmeans.set_areas(a); gm_mesh_kmeans.fit(com); gm_mesh_kmeans.set_covars(None); gm_mesh_kmeans.set_areas(None)
print((time.time()-t1)*1000)


gm_areas_kmeans = GaussianMixture(K,init_params='kmeans',tol=1e-5,max_iter=100);  gm_areas_kmeans.set_areas(a); gm_areas_kmeans.fit(com);  gm_areas_kmeans.set_areas(None)

data_log_mesh = []
data_log_meshk = []
data_log_verts = []
data_log_vertsk = []
data_log_areas = []
data_log_icp = []
data_log_cpd = []
data_log_oracle = []
opt_times = []
opt_times_pts = []
icp_times = []
cpd_times = []

prev_time = time.time()
for n in range(SAMPLE_NUM):

    indices2 = np.random.randint(0,mesh_pts.vertices.shape[0],SAMPLE_PTS)
    samples_for_icp = np.copy(mesh_pts.vertices[indices2])
    full_points = samples_for_icp#mesh_pts.vertices
    indices = np.random.randint(0,mesh_pts.vertices.shape[0],SAMPLE_PTS)
    samples = np.copy(mesh_pts.vertices[indices])
    samples_mean = samples.mean(0)
    centered_points = samples - samples_mean

    print(n,round(time.time()-prev_time,1),'seconds')
    prev_time = time.time()
    if False: # random transformations
        q = np.random.randn(4)
        q = q/np.linalg.norm(q)
        M =  transforms3d.quaternions.quat2mat(q)
        t = np.random.randn(3)*0.05
    else:
        t = np.random.rand(3)*0.1 - 0.05
        angles = np.random.rand(3)*30 - 15
        angles *= np.pi/180.0
        M = transforms3d.euler.euler2mat(angles[0],angles[1],angles[2])

    true_q = transforms3d.quaternions.mat2quat(M)

    source = centered_points @ M + samples_mean+ t
    sourcemean = source.mean(0)
    source_centered = source - sourcemean

    H = (source-source.mean(0)).T @ (samples-samples.mean(0))
    u,s,vt = np.linalg.svd(H)
    R_reg = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
    t_reg = source.mean(0)-samples.mean(0)
    oracle_q = transforms3d.quaternions.mat2quat(R_reg)
    data_log_oracle.append( [oracle_q.dot(true_q),np.linalg.norm(t_reg-t)] )



    def loss_verts(x):
        qs = x[:4]
        ts = x[4:]
        qs = qs/np.linalg.norm(qs)
        Ms = transforms3d.quaternions.quat2mat(qs)
        tpts =  (source_centered) @ Ms.T + sourcemean - ts
        return -gm_std.score(tpts)
    t1 = time.time()
    res = opt.minimize(loss_verts,np.array([1,0,0,0,0,0,0]),method=method)
    opt_times_pts.append(time.time()-t1)
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
        tpts =  (source_centered) @ Ms.T + sourcemean - ts
        return -gm_mesh.score(tpts)
    start_opt = time.time()
    res = opt.minimize(loss_mesh,np.array([1,0,0,0,0,0,0]),method=method)
    end_opt = time.time()
    opt_times.append(end_opt-start_opt)
    rq = res.x[:4]
    rq = rq/np.linalg.norm(rq)
    rt = res.x[4:]
    data_log_mesh.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )
    def loss_mesh_k(x):
        qs = x[:4]
        ts = x[4:]
        qs = qs/np.linalg.norm(qs)
        Ms = transforms3d.quaternions.quat2mat(qs)
        tpts =  (source_centered) @ Ms.T + sourcemean - ts
        return -gm_std_km.score(tpts)
    start_opt = time.time()
    res = opt.minimize(loss_mesh_k,np.array([1,0,0,0,0,0,0]),method=method)
    end_opt = time.time()
    opt_times.append(end_opt-start_opt)
    rq = res.x[:4]
    rq = rq/np.linalg.norm(rq)
    rt = res.x[4:]
    data_log_vertsk.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )

    def loss_areas_k(x):
        qs = x[:4]
        ts = x[4:]
        qs = qs/np.linalg.norm(qs)
        Ms = transforms3d.quaternions.quat2mat(qs)
        tpts =  (source_centered) @ Ms.T + sourcemean - ts
        return -gm_areas_kmeans.score(tpts)
    start_opt = time.time()
    res = opt.minimize(loss_areas_k,np.array([1,0,0,0,0,0,0]),method=method)
    end_opt = time.time()
    rq = res.x[:4]
    rq = rq/np.linalg.norm(rq)
    rt = res.x[4:]
    data_log_areas.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )

    if True:
        def loss_mesh_k2(x):
            qs = x[:4]
            ts = x[4:]
            qs = qs/np.linalg.norm(qs)
            Ms = transforms3d.quaternions.quat2mat(qs)
            tpts =  (source_centered) @ Ms.T + sourcemean - ts
            return -gm_mesh_kmeans.score(tpts)
        start_opt = time.time()
        res = opt.minimize(loss_mesh_k2,np.array([1,0,0,0,0,0,0]),method=method)
        end_opt = time.time()
        rq = res.x[:4]
        rq = rq/np.linalg.norm(rq)
        rt = res.x[4:]
        data_log_meshk.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )
    else:
        def loss_mesh_k2(x):
            qs = x[:3]
            ts = x[3:]
            qs[-1] += 1e-9
            angle = np.linalg.norm(qs)
            axis = qs/angle
            Ms = transforms3d.axangles.axangle2mat(axis,angle)
            tpts =  (source_centered) @ Ms.T + sourcemean - ts
            return -gm_mesh_kmeans.score(tpts)
        start_opt = time.time()
        res = opt.minimize(loss_mesh_k2,np.array([0,0,0,0,0,0]),method=method)
        end_opt = time.time()

        rq = res.x[:3]
        rq[-1] += 1e-9

        angle = np.linalg.norm(rq)
        axis = rq/angle
        rq = transforms3d.quaternions.axangle2quat(axis,angle)
        rt = res.x[3:]
        data_log_meshk.append( [rq.dot(true_q),np.linalg.norm(rt-t)] )
    icp_t = np.zeros(3)
    R = np.identity(3)
    source2 = np.copy(source)
    prev_err = 100000000
    flag = True
    t1 = time.time()
    for icp_iter in range(ICP_ITERS):
        dist = cdist(source2,samples_for_icp)
        sample_idx = np.argmin(dist,1)
        matched_pts = samples_for_icp[sample_idx]
        source2mean = source2.mean(0)
        matchedptsmean = matched_pts.mean(0)
        source2centered = source2-source2mean
        it =  source2mean - matchedptsmean
        if flag:
            idx2 = np.argmin(dist,0)
            matched2 = source2[idx2]
            it = (0.5*it) + 0.5*(matched2.mean(0) - samples_for_icp.mean(0))

        H = (source2centered).T @ (matched_pts-matchedptsmean)
        if flag:
            H2 = (matched2-matched2.mean(0)).T @ (samples_for_icp-samples_for_icp.mean(0))
            H2 *= source2.shape[0]/samples_for_icp.shape[0]
            H = H + H2
        u,s,vt = np.linalg.svd(H)
        rotmat = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
        
        #print(rotmat,'\n',M)
        #print(it,'\n',t)
        source2 = (source2centered) @ rotmat.T + source2mean - it 
        err = np.linalg.norm(source2-matched_pts,axis=1)
        #print(err)
        #print(np.diag(cdist(source2,matched_pts)).mean(),len(matched_pts))
        if np.linalg.norm(err-prev_err) < ICP_THRESH:
            break
        prev_err = err
        icp_t += it
        R = R @ rotmat
        #print(it)
        #print(rotmat)

        icp_q = transforms3d.quaternions.mat2quat(R)
        icp_t = icp_t
    icp_times.append(time.time()-t1)
    data_log_icp.append( [icp_q.dot(true_q),np.linalg.norm(icp_t-t)] )

    t1 = time.time()


    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(source[:,0],source[:,1],source[:,2],label='orig')    
        ax.scatter(samples[:,0],samples[:,1],samples[:,2],label='trans')
        result = (source + icp_t) @ R.T
        ax.scatter(source2[:,0],source2[:,1],source2[:,2],label='registered')
        plt.title(str(icp_q.dot(true_q)) + ' ' + str(np.linalg.norm(icp_t-t)))
        plt.legend()
        plt.show()
print(np.array(opt_times_pts).mean()*1000)
print(np.array(opt_times).mean()*1000)
print(np.array(icp_times).mean()*1000)
print(np.array(cpd_times).mean()*1000)
if len(data_log_verts) > 0 :
    np.savetxt('{}_verts2.csv'.format(tag),np.array(data_log_verts),delimiter=',')
    np.savetxt('{}_vertsk2.csv'.format(tag),np.array(data_log_vertsk),delimiter=',')
    np.savetxt('{}_mesh2.csv'.format(tag),np.array(data_log_mesh),delimiter=',')
    np.savetxt('{}_icp2.csv'.format(tag),np.array(data_log_icp),delimiter=',')
    np.savetxt('{}_cpd2.csv'.format(tag),np.array(data_log_cpd),delimiter=',')
    np.savetxt('{}_oracle2.csv'.format(tag),np.array(data_log_oracle),delimiter=',')
    np.savetxt('{}_meshk2.csv'.format(tag),np.array(data_log_meshk),delimiter=',')
    np.savetxt('{}_areas2.csv'.format(tag),np.array(data_log_areas),delimiter=',')
