import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn_pdf

def compute_gmm(x,w=None,k=2,iter_max=10000,e_tol=1e-6):
    if w is None:
        w = np.ones(x.shape[0])
    mu = x[np.random.choice(x.shape[0],k,replace=False),:]
    sigma = np.array([(x.std()/k)*np.identity(x.shape[1]) for _ in range(k)])
    pi = np.ones(shape=k)/k

    mu_prev = mu.copy()

    for _ in range(iter_max):

        # e-step
        gamma = np.zeros(shape=(x.shape[0],k))
        for i in range(k):
            gamma_i = w * pi[i]*mvn_pdf.pdf(x,mean=mu[i],cov=sigma[i])
            gamma[:,i] = gamma_i
        gamma = gamma/gamma.sum(1,keepdims=True)

        # m-step
        for i in range(k):
            new_mu = np.zeros(x.shape[1])
            for j in range(x.shape[0]):
                new_mu += gamma[j,i] * x[j,:]
            new_mu /= gamma.sum(0)[0]
            mu[i,:] = new_mu
            new_sigma = np.identity(x.shape[1])*e_tol
            for j in range(x.shape[0]):
                xv = x[j,:][:,np.newaxis]
                xm = new_mu[:,np.newaxis]
                xd = xv - xm
                new_sigma += gamma[j,i] * (xd @ xd.T)
            new_sigma /= gamma.sum(0)[0]
            sigma[i,:,:] = new_sigma
        pi = gamma.mean(0)
        if ((mu-mu_prev)**2).sum() < 1e-9:
            break
        mu_prev = mu.copy()
    return mu,sigma,pi