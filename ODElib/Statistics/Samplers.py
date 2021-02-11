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
    #set the random seed for this chain
    np.random.seed(modelframework.random_seed)
    #unpacking parameters
    pnames = modelframework.get_pnames()
    
    reject = set(static_parameters)
    
    ps=modelframework.get_parameters(as_dict=True)
    pname_oldpar = {}#stores old parameters (mapping of pname to value), also implies which parameters should be walking
    for p in ps:
        if p not in reject:
            pname_oldpar[p] = ps[p] #copy the value only
            

    #defining the number of iterations
    iterations = np.arange(1, nits, 1)
    if not burnin:
        burnin = int(nits/2)
    #initial prior
    modcalc = modelframework.integrate(predict_obs=True,as_dataframe=False)
    chi = modelframework.get_chi(modcalc)
    pall = []
    chis=[]
    its=[]
    ars=[]
    acceptance_ratio = []
    #print report and update output
    pits = int(nits/10)
    if print_progress:
        print('a priori error', chi)
        print('iteration; ' 'error; ' 'acceptance ratio')
    for it in iterations:
        for p in pname_oldpar:
            modelframework.parameters[p].rwalk()
        modcalc = modelframework.integrate(predict_obs=True,as_dataframe=False)
        chinew = modelframework.get_chi(modcalc)#calculate goodness of fit
        #priors
        priors_old = np.prod(np.array([modelframework.parameters[p].pdf(pname_oldpar[p]) for p in pname_oldpar]))
        priors_new = np.prod(np.array([modelframework.parameters[p].pdf() for p in pname_oldpar]))
        
        #likelihoods
        #likelihood_old=np.exp(-chi)
        #likelihood_new=np.exp(-chinew)
        
        #likelihood ratio
        likelihooratio= np.exp(-chinew+chi)
        acc = np.exp(np.log(likelihooratio)+np.log(priors_new/priors_old))
        #likelihoods = np.append(likelihoods, chinew)
        if acc > np.random.rand():  # KEY STEP
            chi = chinew
            #storing current parameters as old
            ps=modelframework.get_parameters(as_dict=True)
            for p in pname_oldpar:
                pname_oldpar[p]=ps[p]
            #this iteration was accepted
            ars.append(1)
        else: #if chi gets worse, reassign old parameters
            modelframework.set_parameters(**pname_oldpar)#reassigning parameter values
            #this iteration was rejected
            ars.append(0)

        if it > burnin:  # only store the parameters if you've gone through the burnin period
            pall.append(modelframework.get_parameters(as_dict=True))#stores current parameter set as dictionary
            chis.append(chi)
            its.append(it)
            acceptance_ratio.append(np.array(ars).mean())
            #ar = ar + 1.0  # acceptance ratio
            #ic = ic + 1  # total count
        
    #likelihoods = likelihoods[burnin:]
    #iterations = iterations[burnin:]
    #pall = pall[:,:ic]
    #print_posterior_statistics(pall,pnames)
    df = pd.DataFrame(pall)
    df['chi']=chis
    df['iteration']=its
    df['acceptance_ratio'] = acceptance_ratio
    for p in static_parameters:
        if isinstance(modelframework.parameters[p],np.ndarray):
            df[p] = [modelframework.parameters[p] for el in range(0,len(df))]
        else:
            df[p]=modelframework.parameters[p]
    if df.empty:
        df = pd.DataFrame([[np.nan] * (len(pnames)+3)])
    #df.columns = list(pnames)+['chi','adjR2','Iteration']
    return df