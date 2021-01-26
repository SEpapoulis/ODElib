import numpy as np
import pandas as pd
from pyDOE2 import lhs
from .. import Framework #we need this for isinstance

def sample_lhs(parameter_dict,samples):#,**kwargs):
        '''Sample parameter space using a Latin Hyper Cube sampling scheme

        Parameters
        ----------
        parameter_dict : dict
            parameter names mapped to a ODElib.parameters objects assigned distribuitons (and hyperparameters if applicable)
        samples: int
            number of LHS samples to be taken
        Returns
        -------
        DataFrame
            DataFrame storing each LHS sample, organized into columns by the parameter name. Note that arrays are stored within the rows


        Notes
        -----

        
        '''
        #must enumerate the number of parameters (i.e., total non-zero elements among arrays)
        #TODO: pass array of priors if parameters are stored in an array
        total_ps = 0
        #count all non-zero parameters
        for p in parameter_dict:
            if not isinstance(parameter_dict[p],Framework.parameter):
                raise TypeError()
            nump = np.count_nonzero(parameter_dict[p].val)
            total_ps+=nump
        lhd = lhs(total_ps, samples=samples)#sample in multidimentional space
        var_samples = {}
        lhd_i=0
        for p in parameter_dict:
            nump = np.count_nonzero(parameter_dict[p].val)#count non-zero parameters
            samples = lhd[:,lhd_i:lhd_i+nump]
            lhd_i+=nump
            samples = parameter_dict[p].dist.ppf(samples,**parameter_dict[p].hp)
            if nump == 1:
                var_samples[p] = np.concatenate(samples,axis=None)
            else:
                _sample = []
                _p = parameter_dict[p][0]
                for row in samples:
                    _p[np.where(_p!=0)] = row
                    _sample.append( np.copy(_p) )
                var_samples[p] = _sample
        df = pd.DataFrame(var_samples)
        return(df)

def MetropolisHastings(modelframework,nits=1000,burnin=None,static_parameters=set(),print_progress=True):
    '''allows option to return model solutions at sample times

    Parameters
    ----------
    nits : int
        number of iterations
    burnin : int
        number of iterations to ignore initially, Defaults to half of nits
    static_parameters : list-like, optional
        specify parameters that you do not want to change during the markov chain
    Returns
    -------
    tupple : pall, likelihoods, iterations
        host and virus counts
    '''
    #unpacking parameters
    pnames = modelframework.get_pnames()
    ar,ic = 0.0,0
    ars, likelihoods = np.r_[[]], np.r_[[]]
    
    
    reject = set(static_parameters)
    
    ps=modelframework.get_parameters(asdict=True)
    pname_oldpar = {}#stores old parameters (mapping of pname to value), also implies which parameters should be walking
    for p in ps:
        if p not in reject:
            pname_oldpar[p] = ps[p]
            
    #npars = len(pars)
    #opt = np.ones(npars)
    #stds = np.zeros(npars) + 0.05
    #defining the number of iterations
    iterations = np.arange(1, nits, 1)
    if not burnin:
        burnin = int(nits/2)
    #initial prior
    modcalc = modelframework.integrate(predict_obs=True,as_dataframe=False)
    chi = modelframework.get_chi(modcalc)
    pall = []
    chis=[]
    #print report and update output
    pits = int(nits/10)
    if print_progress:
        print('a priori error', chi)
        print('iteration; ' 'error; ' 'acceptance ratio')
    for it in iterations:
        for p in pname_oldpar:
            modelframework.parameters[p].rwalk()
        modcalc = modelframework.integrate(predict_obs=True,as_dataframe=False)
        chinew = modelframework.get_chi(modcalc)
        
        #test = [p+'='+str(modelframework.parameters[p].val) for p in modelframework.parameters]
        #test = ['chi={}'.format(chinew)]+test
        #print (' '.join(test))

        #likelihoods = np.append(likelihoods, chinew)
        if np.exp(chi-chinew) > np.random.rand():  # KEY STEP
            chi = chinew
            if it > burnin:  # only store the parameters if you've gone through the burnin period
                pall.append(modelframework.get_parameters(asdict=True))#stores current parameter set as dictionary
                chis.append(chi)
                ar = ar + 1.0  # acceptance ratio
                ic = ic + 1  # total count
        else: #if chi gets worse, reassign old parameters
            modelframework.set_parameters(**pname_oldpar)#reassigning parameter values
        if (it % pits == 0) and print_progress:
            print(it,';', round(chi,2),';', ar/pits)
            ars = np.append(ars, ar/pits)
            ar = 0.0
    likelihoods = likelihoods[burnin:]
    iterations = iterations[burnin:]
    #pall = pall[:,:ic]
    #print_posterior_statistics(pall,pnames)
    df = pd.DataFrame(pall)
    df['chi']=chis
    for p in static_parameters:
        if isinstance(modelframework.parameters[p],np.ndarray):
            df[p] = [modelframework.parameters[p] for el in range(0,len(df))]
        else:
            df[p]=modelframework.parameters[p]
    if df.empty:
        df = pd.DataFrame([[np.nan] * (len(pnames)+3)])
    #df.columns = list(pnames)+['chi','adjR2','Iteration']
    return df