import numpy as np

import scipy.stats as stats
from scipy.stats import expon
from scipy import linalg


##=================================================================
def fit(x,y,niter_max=100,l2=0.):
## input: x[L,n], y[L,ny]
## output: h0_all[ny], w_all[ny,n]

    n = x.shape[1]
    ny = y.shape[1]

    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True) #covariance matrix of x
    c += l2*np.identity(n) / (2*len(y))
    c_inv = linalg.pinvh(c)

    w_all = np.zeros((n,ny))
    h0_all = np.zeros(ny)
    cost_all = np.zeros((niter_max,ny))

    for i in range(ny):
        yi = y[:,i]

        # initial values
        h0 = 0.
        w = np.random.normal(0.0,1./np.sqrt(n),size=(n))

        cost = np.full(niter_max,100.)
        for iloop in range(niter_max):
            h = h0 + x.dot(w)
            #y1_model = np.tanh(h) #Binary model
            yi_model = (h*np.cosh(h) - np.sinh(h))/h/np.sinh(h)
            # stopping criterion
            cost[iloop] = ((yi[:]-yi_model[:])**2).mean()        
            #if iloop>0 and cost[iloop] >= cost[iloop-1]: break

            # update local field
            t = h!=0    
            h[t] *= yi[t]/yi_model[t]
            h[~t] = yi[~t]

            # find w from h    
            h_av = h.mean()
            dh = h - h_av 
            dhdx = dh[:,np.newaxis]*dx[:,:]

            dhdx_av = dhdx.mean(axis=0)
            w = c_inv.dot(dhdx_av)
            h0 = h_av - x_av.dot(w)

        # all spins
        h0_all[i] = h0
        w_all[:,i] = w  # Wji: interaction from j to i
        cost_all[:,i] = cost
        
    return h0_all,w_all

##==================================================================
def model_expectation(x,h0,w):
    """
    predict y_model based on x,h0, and w
    input: x[l,n], w[n], h0
    output: y
    """
    h = h0 + x.dot(w)
    y = (h*np.cosh(h) - np.sinh(h))/h/np.sinh(h)
        
    return y

##=========================================================================
def predict(x,h0,w):
## input: x[L,N], h0, w[n]
## output: y[L]

    x_min, x_max = -1., 1.
    
    L,N = x.shape

    y = np.zeros(L)
    #x[0,:] = np.random.rand(1,N) #Initial values of x_i(0)

    #Generating sequences of x_i(t+1)
    for t in range(L):
        h = x[t,:].dot(w) + h0

        if h != 0.:
            x_scale = 1./np.abs(h)
            sampling = stats.truncexpon(b=(x_max-x_min)/x_scale, loc=x_min, scale=x_scale) 
            #truncated exponential dist exp(x h[i]) for x ~ [-1, 1]
            sample = sampling.rvs(1) #obtain 1 samples
            y[t] = -np.sign(h)*sample[0]
        else:
            y[t] = random.uniform(x_min, x_max)
        
    return y

##==================================================================
def generate_data(L,N):
    # L : time steps
    # N : number of nodes
    w_true = np.random.normal(0, 1., (N,N))
    x_min, x_max = -1., 1.
    x = np.zeros([L,N])
    x[0,:] = np.random.rand(1,N) #Initial values of x_i(0)

    #Generating sequences of x_i(t+1)
    for t in range(L-1):
        h = x[t,:].dot(w_true)
        for i in range(N):
            if h[i] != 0.:
                x_scale = 1./np.abs(h[i])
                sampling = stats.truncexpon(b=(x_max-x_min)/x_scale, loc=x_min, scale=x_scale) 
                #truncated exponential dist exp(x h[i]) for x ~ [-1, 1]
                sample = sampling.rvs(1) #obtain 1 samples
                x[t+1, i] = -np.sign(h[i])*sample[0]
            else:
                x[t+1,i] = random.uniform(x_min, x_max)
    return w_true,x

