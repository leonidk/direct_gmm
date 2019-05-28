_tab20c_data = (
    (0.19215686274509805, 0.5098039215686274,  0.7411764705882353  ),  # 3182bd
    (0.4196078431372549,  0.6823529411764706,  0.8392156862745098  ),  # 6baed6
    (0.6196078431372549,  0.792156862745098,   0.8823529411764706  ),  # 9ecae1
    (0.7764705882352941,  0.8588235294117647,  0.9372549019607843  ),  # c6dbef
    (0.9019607843137255,  0.3333333333333333,  0.050980392156862744),  # e6550d
    (0.9921568627450981,  0.5529411764705883,  0.23529411764705882 ),  # fd8d3c
    (0.9921568627450981,  0.6823529411764706,  0.4196078431372549  ),  # fdae6b
    (0.9921568627450981,  0.8156862745098039,  0.6352941176470588  ),  # fdd0a2
    (0.19215686274509805, 0.6392156862745098,  0.32941176470588235 ),  # 31a354
    (0.4549019607843137,  0.7686274509803922,  0.4627450980392157  ),  # 74c476
    (0.6313725490196078,  0.8509803921568627,  0.6078431372549019  ),  # a1d99b
    (0.7803921568627451,  0.9137254901960784,  0.7529411764705882  ),  # c7e9c0
    (0.4588235294117647,  0.4196078431372549,  0.6941176470588235  ),  # 756bb1
    (0.6196078431372549,  0.6039215686274509,  0.7843137254901961  ),  # 9e9ac8
    (0.7372549019607844,  0.7411764705882353,  0.8627450980392157  ),  # bcbddc
    (0.8549019607843137,  0.8549019607843137,  0.9215686274509803  ),  # dadaeb
    (0.38823529411764707, 0.38823529411764707, 0.38823529411764707 ),  # 636363
    (0.5882352941176471,  0.5882352941176471,  0.5882352941176471  ),  # 969696
    (0.7411764705882353,  0.7411764705882353,  0.7411764705882353  ),  # bdbdbd
    (0.8509803921568627,  0.8509803921568627,  0.8509803921568627  ),  # d9d9d9
)

import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

import matplotlib.pyplot as plt
from scipy.special import logsumexp
import mpl_toolkits.mplot3d as m3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

import pymesh



def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1)) #@ np.array([[1,0,0],[0,0,1],[0,-1,0] ])
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    areas /= areas.min()
    areas = areas.reshape((-1,1))
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


mesh0 = pymesh.load_mesh("bunny/bun_zipper_res4.ply")

#pts = mesh0.vertices @ np.array([[1,0,0],[0,0,1],[0,-1,0] ])

pts,a = get_centroids(mesh0)
face_vert = mesh0.vertices[mesh0.faces.reshape(-1),:].reshape((mesh0.faces.shape[0],3,-1)) #@ np.array([[1,0,0],[0,0,1],[0,-1,0] ])
data_covar = get_tri_covar(face_vert)

K = 20
colors = np.array(_tab20c_data)[:K]

np.random.seed(24)

labels = np.zeros((pts.shape[0],K))
labels[np.arange(pts.shape[0]), np.random.randint(0,K,pts.shape[0])] = 1
#labels = np.exp(10*np.random.rand(pts.shape[0],K))
#labels /= labels.sum(1,keepdims=True)

print(labels.max())
for iteration in range(150):
    # m-step
    new_means = []
    new_covars = []
    new_pis = []
    for k in range(K):
        weights = a * labels[:,k:k+1]
        weight_norm = weights.sum()
        new_mean = (weights * pts).sum(0)/weight_norm
        new_means.append(new_mean)

        t = pts - new_mean
        new_covar = (weights/weight_norm * t).T @ t + ((weights/weight_norm).reshape((-1,1,1)) * data_covar).sum(0)
        new_covars.append(new_covar)

        new_pis.append( weight_norm.mean() )
    new_pis = np.array(new_pis)
    new_pis /= new_pis.sum()

    # e-step 
    for k in range(K):
        labels[:,k] = new_pis[k]*mvn_pdf(new_means[k],new_covars[k]).pdf(pts)
    labels /= labels.sum(1,keepdims=True)

    if (iteration % 1) == 0:
        fig = plt.figure(figsize=plt.figaspect(2.0),frameon=False)

        ax = fig.add_subplot(2,1, 1, projection='3d')

        #colors = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in ['CA3542','27646B']]
        #colors = np.array(colors)/255
        #colors = np.array([[1,0,0],[0,0,1]])
        #ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=20,c=labels@ colors)
        res = ax.plot_trisurf(mesh0.vertices[:,0],mesh0.vertices[:,1],mesh0.vertices[:,2],triangles=mesh0.faces,facecolors=labels@ colors)
        normals = ax._generate_normals(face_vert)
        colset = ax._shade_colors(labels@ colors, normals)
        res.set_facecolors(colset)

        r = max(pts.max(1) - pts.min(1))/2
        m = pts.mean(1)

        ax.set_xlim(m[0]-r,m[0]+r)
        ax.set_xlim(m[1]-r,m[1]+r)
        ax.set_xlim(m[2]-r,m[2]+r)
        ax.view_init(100,-90)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        #ax.set_aspect('equal', 'box')

        plt.title('E-Step Result',size=24,weight='demibold')
        plt.tight_layout()

        ax = fig.add_subplot(2,1, 2, projection='3d')

        for k in range(len(new_means)):
            mean,covar = new_means[k],new_covars[k]
            u,s,vt = np.linalg.svd(covar)
            coefs = (.002, .002, .002)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
            # Radii corresponding to the coefficients:
            rx, ry, rz = 1.7*np.sqrt(s)#s#1/np.sqrt(coefs)
            
            R_reg = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
            
            #print(eigs)
            # Set of all spherical angles:
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)

            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = rx * np.outer(np.cos(u), np.sin(v)) #+ mean[0]
            y = ry * np.outer(np.sin(u), np.sin(v)) #+ mean[1]
            z = rz * np.outer(np.ones_like(u), np.cos(v)) #+ mean[2]
            
            for i in range(len(x)):
                for j in range(len(x)):
                    x[i,j],y[i,j],z[i,j] = np.dot([x[i,j],y[i,j],z[i,j]], vt) + mean    
            # Plot:
            res = ax.plot_surface(x,y,z,  color=colors[k],shade=True,linewidth=0.0,alpha=min(0.5,new_pis[k]*K))
        ax.set_xlim(m[0]-r,m[0]+r)
        ax.set_xlim(m[1]-r,m[1]+r)
        ax.set_xlim(m[2]-r,m[2]+r)
        ax.view_init(100,-90)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        #ax.set_aspect('equal', 'box')


        plt.title('M-Step Result',size=24,weight='demibold')
        plt.tight_layout()
        #plt.show()
        plt.savefig('output2/{:02d}.png'.format(iteration),dpi=300,pad_inches=0)
        plt.close('all')