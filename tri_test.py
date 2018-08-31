import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
from scipy.stats import multivariate_normal as mvg



import random

def point_on_triangle(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    s, t = sorted([random.random(), random.random()])
    return (s * pt1[0] + (t-s)*pt2[0] + (1-t)*pt3[0],
            s * pt1[1] + (t-s)*pt2[1] + (1-t)*pt3[1],
            s * pt1[2] + (t-s)*pt2[2] + (1-t)*pt3[2])

ax = m3d.Axes3D(plt.figure())

vtx = np.array(((0.0,0.0,0.0),(0.0,0.0,2.0),(0.0,1.0,0.0)))
vtx = np.random.randn(3,3)
tri = m3d.art3d.Poly3DCollection([vtx])
tri.set_facecolor((0.7,0.7,1.0,0.5))
tri.set_edgecolor('k')
tri.set_alpha(0.1)

ax.add_collection3d(tri)

mu = vtx.mean(0)
mu =  np.random.randn(3,3)[0,:]/3
covar = np.random.randn(3,3)/2
covar = np.abs(covar.T.dot(covar))
#covar = np.identity(3)
A = vtx[0,:]
B = vtx[1,:]
C = vtx[2,:]
M = (A+B+C)/3.0
s = np.random.multivariate_normal(mu,covar,200)
ax.scatter(s[:,0],s[:,1],s[:,2])
for sn in range(1,101,10):
    x = np.linspace(0,1,sn)
    y = np.linspace(0,1,sn)


    pts = []
    if False:
        for i in range(len(x)):
            for j in range(i,len(y)):
                u = x[i]
                v = 1-y[j]
                #pt = (1-u) * A + u * ((1-v)*B + v*C)
                pt = A + u*(B-A) + v*(C-A)
                pts.append(pt)
    else:
        #nrm1 = np.linalg.norm(B-A)
        #nrm2 = np.linalg.norm(C-A)
        #bound = max(nrm1,nrm2)
        #pts1 = np.random.uniform(-bound,bound,(sn,2))
        pts =  [point_on_triangle(A, B, C) for _ in range(sn)]

        #print(pts1)
        #aise
    pts = np.array(pts)
    #ax.scatter(pts[:,0],pts[:,1],pts[:,2])

    #cov = np.vstack(((B-A),(C-A))).T
    #js = np.sqrt(np.linalg.det(cov.T.dot(cov)))

    at = np.linalg.norm(np.cross((B-A),(C-A)))/2
    #print('TRUTH:\t',at)
    #print('JAC/2:\t',js/2)

    #u = np.random.randn(3,3)[0,:]/3
    #covar = np.random.randn(3,3)/2
    #covar = np.abs(covar.T.dot(covar))


    #l2 = np.exp( np.log() )

    lklh_add = 0.0
    lklh_mul = 0.0
    mll_mul = 0.0
    #        dev = x - mean
    #        maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
    #        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)
    def my_ll(x,u,s):
        dev = x - u
        dev = dev.reshape((3,1))
        res = 0.0
        res -= 0.5 * np.log(2*np.pi) *3 
        res -= 0.5 * np.log(np.linalg.det(s))
        #res -= 0.5 * dev.reshape((1,-1)).dot(np.linalg.inv(s) ).dot(dev.reshape((-1,1)))
        res -= 0.5 * dev.T.dot(np.linalg.inv(s)).dot(dev)

        #res = -0.5 * ()
        return res.sum()
    def tri_ll(A,B,C,mu,s):
        m = ((A+B+C)/3 )
        dev = (m - mu).reshape((-1,1))
        a = A.reshape((-1,1))
        b = B.reshape((-1,1))
        c = C.reshape((-1,1))
        m = m.reshape((-1,1))

        res = 0.0
        res -= at*0.5 * np.log(2*np.pi) *3
        #print(res)
        res -= at*0.5 * np.log(np.linalg.det(s))
        t1 = dev.dot(dev.T)
        t2 = (a.dot(a.T) + b.dot(b.T) + c.dot(c.T) - 3*m.dot(m.T))
        #print(res,at,'\n',t1,'\n',t2/12.0)
        res -= 0.5 *at * np.trace(( t1 + (1/12.0) * t2).dot(np.linalg.inv(s)))
        #print(t2.shape)
        #res = -0.5 * ()
        return res.sum()/at,t2
    def tri_ll2(A,B,C,mu,s):
        m = ((A+B+C)/3 )
        dev = (m - mu).reshape((-1,1))
        a = A.reshape((-1,1))
        b = B.reshape((-1,1))
        c = C.reshape((-1,1))
        m = m.reshape((-1,1))

        res = 0.0
        res -= at*0.5 * np.log(2*np.pi) *3
        #print(res)
        res -= at*0.5 * np.log(np.linalg.det(s))
        prec =  np.linalg.inv(s)
        t1 = dev.dot(dev.T)
        t2 = (a.dot(a.T) + b.dot(b.T) + c.dot(c.T) - 3*m.dot(m.T))
        t22 = (a.T.dot(prec).dot(a) + b.T.dot(prec).dot(b) + c.T.dot(prec).dot(c) - 3*m.T.dot(prec).dot(m))
        #np.trace(( t1 + (1/12.0) * t2).dot(np.linalg.inv(s)))
        #print(res,at,'\n',t1,'\n',t2/12.0)
        res -= 0.5 *at * ((dev.T.dot(np.linalg.inv(s))).dot(dev) + (1.0/12.0)*(t22))
        #print(t2.shape)
        #res = -0.5 * ()
        return res.sum()/at,t2
    l1 = at*mvg.pdf((A+B+C)/3.0,mu,covar)
    est_covar  = np.identity(3)*0
    for p in pts:
        lklh_add += mvg.pdf(p,mu,covar)
        lklh_mul += mvg.logpdf(p,mu,covar)#np.log(mvg.pdf(p,u,covar))
        mll_mul += my_ll(p,mu,covar)
        d = (p-M).reshape((-1,1))
        est_covar += (d.dot(d.T))
    
    est_covar *= (1.0/len(pts))
    ll,cv = tri_ll(A,B,C,mu,covar)
    print(len(pts))
    #print("p(m)A",np.log(l1))
    #print("S p(m)",np.log(lklh_add))
    #print("P p(m)",lklh_mul)
    #print("P p(m2",lklh_mul+(1/len(pts)))
    print("P p(m3",lklh_mul*(1.0/len(pts)))
    #print("constat",np.log(len(pts)))
    #print("P p(m4",lklh_mul/(1/len(pts)))
    #print("P p(m5",lklh_mul-np.log(1/len(pts)))
    #print("P p(m6",lklh_mul*np.log(1/len(pts)))
    #print("P p(m7",lklh_mul/np.log(1/len(pts)))
    #print("M p(m)",mll_mul)
    #print("M p(m3",mll_mul*(1.0/len(pts)))

    print("T p(m)2",tri_ll2(A,B,C,mu,covar)[0])

    print("T p(m)",ll)
    #print("R1p(m)",(ll)/(at*mll_mul*(1.0/len(pts))))

    #print("R2p(m)",(ll/at)/(at*mll_mul*(1.0/len(pts))))
    #print("D1p(m)",(ll) - (at*mll_mul*(1.0/len(pts))))

    #print("D2p(m)",(ll/at)- (at*mll_mul*(1.0/len(pts))))
    #print("covar\n",cv/12.0)
    #print("est covar\n",est_covar)
    #print(np.linalg.norm(cv/12.0 - est_covar))

m = ((A+B+C)/3 )
dev = (m - mu).reshape((-1,1))
a = A.reshape((-1,1))
b = B.reshape((-1,1))
c = C.reshape((-1,1))
m = m.reshape((-1,1))





mu2 = (A+B+C)/3
t1 = dev.dot(dev.T)

t2 = (a.dot(a.T) + b.dot(b.T) + c.dot(c.T) - 3*m.dot(m.T))
covar2 = (  (1/12.0) * t2)
s2 = np.random.multivariate_normal(mu2,covar2,2000)
#print("covar2\n",covar2)
#print("ratio\n",covar2/est_covar)

#ax.scatter(s2[:,0],s2[:,1],s2[:,2])
#plt.show()
