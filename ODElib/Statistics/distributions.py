from scipy.stats import truncnorm
import scipy
import numpy as np

def Positive_Normal(loc,scale):
    '''normal distribution for positive values only'''
    mu = loc
    sigma = scale
    lower = 0
    upper = mu + sigma*100 #essentially not bound
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    dist = truncnorm(a,b,loc=mu,scale=sigma)
    return(dist)

class discrete_norm(scipy.stats.rv_discrete):
    "Normal distribution"
    def _pmf(self,k, mu, sigma):
        return 1/(sigma*(2*np.pi)**0.5)*np.exp(-.5*((k-mu)/sigma)**2)


class gamma_gen(scipy.stats.rv_continuous):
    '''Gamma Distribution'''
    def _pdf(self,x,alpha,ref):
        A=alpha
        B=ref/alpha
        G = scipy.special.gamma(A)
        return( 1/(B**A*G)**(x**(A-1)*np.exp(-x/B)) )


gamma = gamma_gen(name='Gamma Distribution')