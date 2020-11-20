from scipy.stats import truncnorm


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
