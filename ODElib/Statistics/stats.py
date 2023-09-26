import numpy as np

def predict_logsigma(sigma,mean):
    '''
    This function predicts the log transformed standard deviation from
    the mean and standard devation of untransformed data

    Parameters
    ----------
    sigma : numpy.ndarray
        standard deviations, calculated without transformations
    mean : numpy.ndarray
        mean, calculated without transformation

    Returns
    -------
    numpy.ndarray
        an array containing the variance as if it was calculated in log space
    '''
    return(np.log(1.0+sigma**2.0/mean**2.0)**0.5)

def chi(O,C,S):
    '''calculate reduced chi squared
    
    Parameters
    ----------
    O : numpy.ndarray
        observed values
    C : numpy.ndarray
        calculated values
    S : numpy.ndarray
        variance

    Returns:
    -------
    chi
        fit of calculated values, lower values indicate a better fit
    
    '''
    return( ( (np.ma.masked_invalid(O)-C)**2 / (2*(S**2)) ).sum() )


def AIC(chi,num_parameters):
    '''calcualte Akaike information criterion (AIC) for the model fit'''
    AIC = -2*(-chi) + 2*num_parameters
    return(AIC)

def Rsqrd(C_dict,O_dict):
    '''calculate R^2'''
    sstot=0
    ssres=0
    for sname in C_dict:
        ssres += np.nansum((C_dict[sname]-O_dict[sname])**2)
        sstot += C_dict[sname].shape[0]*np.var(O_dict[sname])
    return (1 - ssres / sstot)

def get_adjusted_rsquared(Rsqrd,num_samples,num_parameters):
    '''calculate adjusted R^2'''
    n = num_samples
    p = num_parameters
    adjR2 = 1 - (1-Rsqrd)*(n-1)/(n-p-1)
    return adjR2
