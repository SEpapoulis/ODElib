import pandas as pd
import numpy as np
from scipy.integrate import odeint
import multiprocessing
from scipy.stats import truncnorm,norm,uniform
from pyDOE2 import lhs
from . import distributions


def rawstats(pdseries):
    '''calculates raw median and standard deviation of posterior'''
    log_mean = np.log(pdseries).mean()
    median = np.exp(log_mean)
    log_std = np.log(pdseries).std()
    std = ((np.exp(log_std**2)-1)*np.exp(2*log_mean+log_std**2.0))**0.5
    return(median,std)

def Chain_worker(model,argdict):
    '''Function called by pool for parallized fitting'''

    posterior = model._MarkovChain(**argdict)
    return(posterior)

def Integrate_worker(model,parameters=None,only_timefinal=False):
    ps = None
    if parameters:
        ps = model.get_parameters(**parameters)
    mod = model.integrate(parameters=ps)
    if parameters:
        for p in parameters:
            mod[p] = [parameters[p]]*len(mod)
    if only_timefinal:
        return (mod.iloc[-1])
    return(mod)

def Fit_worker(model,parameters):
    mod = model.integrate(parameters=parameters,forshow=False)
    chi = mod.get_chi(mod)
    ps = list(parameters[0])
    ps.append(chi)
    return(ps)

def predict_logsigma(sigma,mean):
    '''
    This function predicts the log transformed standard deviation from
    the mean and standard devation of untransformed data
    '''
    s = np.log(1.0+(sigma**2.0/mean**2.0))**0.5
    return(s)

class ModelFramework():

    def __init__(self,ODE,parameter_names=None,state_names=None,dataframe=None,state_summations=None,
                t_end=5,t_steps=1000,random_seed=0,**kwargs):
        '''
        The SnI (Susceptible n-Infected) class acts a framework to facilitate and expedite the analysis
         of viral host interactions. Specifically, this class uses a Markov Chain Monte Carlo (MCMC) 
         implementation to fit and generate posterior distributions of those parameters. Several 
         functions have been included to provide arguments for scipy.integrate.odeint

        Parameters
        ----------
        dataframe : pandas.DataFrame
            A dataframe indexed by organism with fields time, abundance, and uncertainty
        Infection_states : int
            Number of infected states of host
        '''
        
        '''
        self.df = self._df_check(dataframe)

        #time steps for numerical integration
        days=max(np.amax(self.df.loc['virus']['time']),np.amax(self.df.loc['host']['time']))
        self.times = np.arange(0, days, 900.0 / 86400.0) 
        
        #indexes to retireve from numerical integration results that match times of data in df
        self._pred_h_index = np.r_[[np.where(abs(a-self.times) == min(abs(a-self.times)))[0][0] for a in self.df.loc['host']['time']]]
        self._pred_v_index = np.r_[[np.where(abs(a-self.times) == min(abs(a-self.times)))[0][0] for a in self.df.loc['virus']['time']]]
        #stored values for R2 and chi calculations
        self._ha = np.array(self.df.loc['host']['abundance'])
        self._hu = np.array(self.df.loc['host']['uncertainty'])
        self._va = np.array(self.df.loc['virus']['abundance'])
        self._vu = np.array(self.df.loc['virus']['uncertainty'])
        '''
        _is = {}#initial states
        self.random_seed = random_seed
        np.random.seed(random_seed)
        #parameter assignment
        #pnames is referenced by multilpe functions
        if parameter_names:
            self._pnames = parameter_names
        else:
            self._pnames = kwargs['parameter_names']
        if state_names:
            self._snames = state_names
        else:
            self._snames = kwargs['state_names']
        
        self.parameters = {el:None for el in self._pnames}
        self.istates = {el:None for el in self._snames} #initial states
        
        _ps = {} #parameters
        for el in kwargs:
            if el in self._pnames:
                _ps[el] = kwargs[el]
            if el in self._snames:
                _is[el] = kwargs[el] #overiding dataframe initial states
        self.set_parameters(**_ps)
        
        self._model = ODE
        
        self._state_summations = state_summations
        if self._state_summations:
            self._summations_index = self._get_summation_index()
        else:
            self._summations_index = None
        #self._samples = self.df.loc['host']['abundance'].shape[0] + self.df.loc['virus']['abundance'].shape[0]

        self.df = None

        if not isinstance(dataframe,type(None)):
            self.df = self._processdf(dataframe)
            #self.times = np.arange(0, max(self.df['time']), max(self.df['time'])/t_step) 
            self.times = np.linspace(0, max(self.df['time']),t_steps)
            self._pred_tindex = {} #stores time index for predicted values
            for pred in set(self.df.index):
                if isinstance(self.df.loc[pred]['time'],pd.core.series.Series):
                    self._pred_tindex[pred] = np.r_[[np.where(abs(a-self.times) == min(abs(a-self.times)))[0][0] for a in self.df.loc[pred]['time']]]
                else:
                    a = self.df.loc[pred]['time']
                    self._pred_tindex[pred] = np.r_[np.where(abs(a-self.times) == min(abs(a-self.times)))[0][0]]
            #setting the inital values
            for org,log_abundance in self.df[self.df['time'] == 0]['log_abundance'].iteritems():
                if org not in _is:
                    _is[org] = np.exp(log_abundance)# SETTING INIT TO MEDIAN, OR MEAN IN LOG!
        else:
            self.times = np.linspace(0, t_end, t_steps)
        
        self.set_inits(**_is)

    def _processdf(self,df):
        if 'replicate' in df:
            df = df[['time','abundance']]
            df['log_abundance'] = np.log(df['abundance'])
            dfagg = df.groupby(by=['time','organism']).mean()
            dfagg['log_sigma'] = df.groupby(by=['time','organism']).std()['log_abundance']
            dfagg.reset_index('time',inplace=True)
        else:
            if 'log_abundance' not in df:
                df['log_abundance'] = np.log(df['abundance'])
            if 'log_sigma' not in df:
                df['log_sigma'] = predict_logsigma(df['sigma'],df['abundance'])
            dfagg=df
        return(dfagg)

    def _get_summation_index(self):
        sumpop_sumi = {}
        for sumpop in self._state_summations:
            _t = []
            for el in self._snames:
                if el in self._state_summations[sumpop]:
                    _t.append(True)
                else:
                    _t.append(False)
            sumpop_sumi[sumpop] = _t
        return(sumpop_sumi)

    def get_pnames(self):
        '''returns the names of the parameters used in the current model'''
        return(self._pnames)

    def get_snames(self,after_summation=False,predict_df=False):
        '''returns the names of the state variables used in the current model'''
        if after_summation and self._state_summations:
            snames = []
            summed_pops = []
            exclude = set()
            for summed_pop in self._state_summations:
                exclude = exclude.union(set(self._state_summations[summed_pop]))
                summed_pops.append(summed_pop)
            for pop in self._snames:
                if pop not in exclude:
                    snames.append(pop)
            snames.extend(summed_pops)
            return(snames)
        elif predict_df:
            return(list(self._pred_tindex.keys()))
        else:
            return(self._snames)


    def __repr__(self):
        '''pretty printing'''
        outstr = ["Current Model = {}".format(str(self._model.__module__)+'.'+str(self._model.__name__)),
                "Parameters:",
        ]
        for p in self.get_pnames():
            outstr.append("\t{} = {}".format(p,self.parameters[p]))
        outstr.append("Initial States:")
        for s in self.get_snames():
            outstr.append("\t{} = {}".format(s,self.istates[s]))
        return('\n'.join(outstr))

    def __str__(self):
        return(self.__repr__())

    def set_parameters(self,**kwargs):
        '''set parameters for the model
        
        Parameters
        ----------
        **kwargs
            key word arguments, where keys are parameters and args are parameter values. Alternativly, pass **dict
        '''
        pset = set(self._pnames) #sets are faster when checking membership!
        for p in kwargs:
            if p in pset:
                self.parameters[p] = kwargs[p]
            else:
                raise Exception("{} is an unknown parameter. Acceptable parameters are: {}".format(p,', '.join(self._pnames)))

    def set_inits(self,**kwargs):
        '''set parameters for the model
        
        Parameters
        ----------
        **kwargs
            key word arguments, where keys are parameters and args are parameter values. Alternativly, pass **dict
        '''
        sset = set(self._snames) #sets are faster when checking membership!
        for s in kwargs:
            if s in sset:
                self.istates[s] = kwargs[s]
            else:
                raise Exception("{} is an unknown state variable. Acceptable parameters are: {}".format(s,', '.join(self._snames)))

    def get_inits(self,as_dict=False):
        if as_dict:
            return(self.istates)
        inits = np.array([self.istates[el] for el in self.get_snames()])
        return(inits)
    
    def find_inits(self,var_dist=dict(),set_best=True,step=1,**kwargs):
        '''get the initial state variable values for integration

        Parameters
        ----------
        var_dist : tulple, optional
            a mapping of state varaible names to a tuple of a scipy distribution and a boolean. If the boolean is true,
            samples drawn from the specified distribution will be exponentiated 
        virus_init : int, optional
            ignore v0 in data and set the viral initial value

        Return
        ------
        numpy array
            a numpy array of initial values for integration
        '''
        #default_dist = (uniform(loc=0,scale=10),True)
        missing = set(self.get_snames()) - set(var_dist.keys()).union(set(kwargs.keys()))
        if missing:
            raise ValueError("Distributions or specific values were not provided for {}".format(', '.join(missing)))
        
        inits = self._lhs_samples(var_dist,samples=10000,**kwargs) #inits is a dataframe
        
        ps = self.get_parameters()
        results = []
        for row in inits[self.get_snames()].itertuples(index=False):
            row = np.array(row)
            d = self._model(y=row,t=step,ps=ps[0]) #returns differential
            dlog = np.log(d)
            if not np.any(np.isnan(dlog)):
                results.append(np.r_[row,dlog.sum()])
        df = pd.DataFrame(results)
        return(df)

    def get_model(self):
        return(self._model)

    def get_parameters(self,asdict=False,**kwargs):
        '''return the parameters needed for integration
        
        Parameters
        ----------
        asdict : bool, optional
            If true, return dict with parameter names mapped to values
        kwargs: optional
            pass a mapping of parameters to be packages for value return
        Return
        ------
        parameters
            numpy array of parameters ready for odeint or dict of parameters
        '''
        if asdict:
            ps = {}
            for p in self.get_pnames():
                if p in kwargs:
                    ps[p] = kwargs[p]
                else:
                    ps[p] = self.parameters[p]
        else:
            ps = []
            for p in self.get_pnames():
                if p in kwargs:
                    ps.append(kwargs[p])
                else:    
                    ps.append(self.parameters[p])
            ps = tuple([ps])
        return(ps)

    def get_numstatevar(self):
        '''returns the number of state varaibles'''
        return(len(self._snames))

    def _lhs_samples(self,var_mapping,samples):#,**kwargs):
        '''Sample parameter space using a Latin Hyper Cube sampling scheme

        Parameters
        ----------
        **kwargs
            keyword arguments, where key words are mapped to a tuple of mean, sigma, bool
            for if the parameter can only be positive, and a bool for if the draws should
            be negativly exponentiated 
        '''
        #must enumerate the number of parameters (i.e., total non-zero elements among arrays)
        #TODO: pass array of priors if parameters are stored in an array
        total_ps = 0
        for p in var_mapping:
            nump = np.count_nonzero(self.parameters[p])
            total_ps+=nump
        lhd = lhs(total_ps, samples=samples)
        var_samples = {}
        lhd_i=0
        for p in var_mapping:
            nump = np.count_nonzero(self.parameters[p])
            samples = lhd[:,lhd_i:lhd_i+nump]
            lhd_i+=nump
            dist,lambda_trans=var_mapping[p]
            samples = dist.ppf(samples)
            if lambda_trans:
                samples = lambda_trans(samples)
            if nump == 1:
                var_samples[p] = samples
            else:
                _sample = []
                _p = self.parameters[p]
                for row in samples:
                    _p[np.where(_p!=0)] = row
                    _sample.append( np.copy(_p) )
                var_samples[p] = _sample

        df = pd.DataFrame(var_samples)
        return(df)

    def integrate(self,inits=None,parameters=None,predict_df=False,as_dataframe=True,sum_subpopulations=True):
        '''allows option to return model solutions at sample times

        Parameters
        ----------
        inits : numpy.array, optional
            ignore h0 and v0 in data and set the initial values for integration
        parameters : numpy.array, optional
            ignore stored parameters and use specified
        predict_df : bool
            If true, only time points in df will be returned

        Returns
        -------
        tupple : (h, v)
            host and virus counts
        '''
        
        func = self.get_model()
        if isinstance(inits,type(None)):
            initials=list(self.get_inits())
        else:
            initials = inits
        if not parameters:
            ps = self.get_parameters()
        else:
            ps = parameters
        mod = odeint(func,initials,self.times,args=ps)

        #subpopulation summations
        if sum_subpopulations and self._summations_index:
            summed = []
            keep = None
            for sumpop in self._summations_index:
                summed.append(mod[:,self._summations_index[sumpop]].sum(axis=1))
                if not keep:
                    keep = [not el for el in self._summations_index[sumpop]]
                else:
                    keep = [el1 and (not el2) for el1,el2 in zip(keep,self._summations_index[sumpop])]
            
            mod = np.concatenate([mod[:,keep],np.array(summed).T],axis=1)

        if as_dataframe or predict_df:
            df = pd.DataFrame(mod)
            df.columns = self.get_snames(after_summation=sum_subpopulations)
            df['time'] = self.times
            if predict_df:
                calc=pd.melt(df[self.get_snames(predict_df=True)+['time']],id_vars=['time'])
                calc.columns = ['time','organism','abundance']
                calc=calc.set_index('organism')
                return(pd.concat([calc.loc[sname].iloc[self._pred_tindex[sname]] for sname in self.get_snames(predict_df=True)]))                
            return(df)
        return mod


    def get_chi(self,O,C,sigma):
        '''calculate chi values from predicted values'''
        #chi = sum((np.log(h) - np.log(self._ha)) ** 2 / \
        #            (np.log(1.0+self._hu**2.0/self._ha**2.0)**0.5 ** 2)
        #            ) \
        #    + sum((np.log(v) - np.log(self._va)) ** 2 / \
        #            (np.log(1.0+self._vu**2.0/self._va**2.0)**0.5 ** 2)
        #            )
        chi = np.sum( (O - C)**2 / sigma**2 )
        return(chi)

    def get_rsquared(self,h,v):
        '''calculate R^2'''
        ssres = sum((h - self._ha) ** 2) \
            + sum((v - self._va) ** 2)
        sstot = h.shape[0]*np.var(self._ha) \
            + v.shape[0]*np.var(self._va)
        return 1 - ssres / sstot


    def get_adjusted_rsquared(self,h,v):
        '''calculate adjusted R^2'''
        R2 = self.get_rsquared(h,v)
        n = self._samples
        p = len(self.get_pnames())
        adjR2 = 1 - (1-R2)*(n-1)/(n-p-1)
        return adjR2

    def get_AIC(self,chi):
        '''calcualte AIC for the model fit'''
        K = len(self.get_pnames())
        AIC = -2*np.log(np.exp(-chi)) + 2*K
        return(AIC)

    def get_fitstats(self,h=None,v=None):
        '''return dictionary of adjusted R-squared, Chi, and AIC of current parameters'''
        fs = {}
        if isinstance(h,type(None)) or isinstance(v,type(None)):
            h,v = self.integrate(predict_df=True)
        fs['Chi'] = self.get_chi(h,v)
        fs['AdjR^2'] = self.get_adjusted_rsquared(h,v)
        fs['AIC'] = self.get_AIC(fs['Chi'])
        return(fs)
    
    def _rand_walk(self,pdict):
        _pdict={}
        stds=.05
        for p in pdict:
            if isinstance(pdict[p],np.ndarray):
                _pdict[p]=np.exp(np.log(pdict[p]) + np.random.normal(0, stds,pdict[p].shape))
            else:
                _pdict[p]=np.exp(np.log(pdict[p]) + np.random.normal(0, stds))
        return(_pdict) 

    def _MarkovChain(self,nits=1000,burnin=None,static_parameters=None,print_progress=True):
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
        pnames = self.get_pnames()
        ar,ic = 0.0,0
        ars, likelihoods = np.r_[[]], np.r_[[]]
        pars = {}
        if static_parameters:
            reject = set(static_parameters)
        else:
            reject = set()
        ps=self.get_parameters(asdict=True)
        for p in ps:
            if p not in reject:
                pars[p] = ps[p]
                #try:
                #    pars[p] = np.float(ps[p])#we need to enforce dtype for computation to work
                #except TypeError:
                #    pars[p] = ps[p]
        npars = len(pars)
        opt = np.ones(npars)
        stds = np.zeros(npars) + 0.05
        #defining the number of iterations
        iterations = np.arange(1, nits, 1)
        if not burnin:
            burnin = int(nits/2)
        #initial prior
        modcalc = self.integrate(predict_df=True,parameters = self.get_parameters(**pars))
        chi = np.sum([self.get_chi(O=self.df.loc[sname]['log_abundance'],
                                    C=np.log(modcalc.loc[sname]['abundance']),
                                    sigma=self.df.loc[sname]['log_sigma'])
                    for sname in self.get_snames(predict_df=True)])
        pall = []
        chis=[]
        #print report and update output
        pits = int(nits/10)
        if print_progress:
            print('a priori error', chi)
            print('iteration; ' 'error; ' 'acceptance ratio')
        for it in iterations:
            pars_old = pars
            pars = self._rand_walk(pars)#permit
            modcalc = self.integrate(parameters = self.get_parameters(**pars),predict_df=True)
            chinew = np.sum([self.get_chi(O=self.df.loc[sname]['log_abundance'],
                                            C=np.log(modcalc.loc[sname]['abundance']),
                                            sigma=self.df.loc[sname]['log_sigma'])
                            for sname in self.get_snames(predict_df=True)])
            likelihoods = np.append(likelihoods, chinew)
            if np.exp(chi-chinew) > np.random.rand():  # KEY STEP
                chi = chinew
                if it > burnin:  # only store the parameters if you've gone through the burnin period
                    pall.append(pars)
                    chis.append(chi)
                    ar = ar + 1.0  # acceptance ratio
                    ic = ic + 1  # total count
            else: #if chi gets worse, use old parameters
                pars = pars_old
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
            if isinstance(self.parameters[p],np.ndarray):
                df[p] = [self.parameters[p] for el in range(0,len(df))]
            else:
                df[p]=self.parameters[p]
        if df.empty:
            df = pd.DataFrame([[np.nan] * (len(pnames)+3)])
        #df.columns = list(pnames)+['chi','adjR2','Iteration']
        return df

    def _parallelize(self,func,args,cores):
        '''Parallelized Markov Chains
        
        Parameters
        ----------
        func : func
            Function to parallelize
        args : list of lists
            Arguments to be passed to func
        cores : int
            number of cpu cores to use

        
        Returns
        -------
        list
            list of outputs from func
        
        '''
        if cores > multiprocessing.cpu_count():
            Warning("More cores specified than avalible, cpu_cores set to maximum avalible\n")
            cores=multiprocessing.cpu_count()
        print("Starting {} processes with {} cores".format(len(args),cores),end='\r')
        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.starmap(func,args)
        pool.join()
        pool.close()
        print("Starting {} processes with {} cores\t[DONE]".format(len(args),cores))
        return(results)

    def explore_paramspace(self,samples=1000,cpu_cores=1,aggregate_endpoints=True,**kwargs):
        '''search parameter space for good initial parameter values

        Parameters
        ----------
        samples : int
            Number of samples to search
        cpu_cores : int
            number of cpu cores to use
        **kwargs
            parameters mapped to tuples. Tuples should include three values: mean, standard deviation, and a boolean for
            tinylog transformation. Tinylog transformation is defined as `np.power(10,-(pos_norm(loc=mu,scale=sigma)`.
            Otherwise, only a pos_norm distribution is sampled, where pos_norm is a normal distribution with the lower
            bound always truncated at zero.

        
        Returns
        -------
        list
            list of outputs from func

        '''
        print("Sampling with a Latin Hypercube scheme")
        ps = self._lhs_samples(kwargs,samples)
        jobs=[]
        for i,row in ps.iterrows():
            mod = self.copy()
            jobs.append([mod,row.to_dict(),aggregate_endpoints])
        #packaging SIn instances with different parameters from LHS sampling
        #args = [[self,tuple((tuple(ps),))] for ps in inits[self.get_pnames()].itertuples(index=False)]
        if cpu_cores ==1:
            results = []
            for job in jobs:
                results.append(Integrate_worker(job[0],job[1],job[2]))
        else:                
            results = self._parallelize(Integrate_worker,jobs,cores=cpu_cores)
        if aggregate_endpoints:
            results = pd.DataFrame(results)
        return(results)
        #return(df)

    #BROKEN
    def search_initparamfits(self,samples=1000,cpu_cores=1,**kwargs):
        '''search parameter space for good initial parameter values

        Parameters
        ----------
        samples : int
            Number of samples to search
        cpu_cores : int
            number of cpu cores to use
        **kwargs
            parameters mapped to tuples. Tuples should include three values: mean, standard deviation, and a boolean for
            tinylog transformation. Tinylog transformation is defined as `np.power(10,-(pos_norm(loc=mu,scale=sigma)`.
            Otherwise, only a pos_norm distribution is sampled, where pos_norm is a normal distribution with the lower
            bound always truncated at zero.

        
        Returns
        -------
        list
            list of outputs from func

        '''
        print("Sampling with a Latin Hypercube scheme")
        inits = self._lhs_samples(samples,**kwargs)
        #packaging SIn instances with different parameters from LHS sampling
        args = [[self,tuple((tuple(ps),))] for ps in inits[self.get_pnames()].itertuples(index=False)]
        if cpu_cores ==1:
            results = []
            for arg in args:
                results.append(Fit_worker(arg[0],arg[1]))
        else:                
            results = self._parallelize(Fit_worker,args,cores=cpu_cores)
        df=pd.DataFrame(results)
        df.columns = self.get_pnames()+['chi']
        df.dropna(inplace=True)
        return(df)

    def _arg_copy(self):
        args = {}
        args['parameter_names']=self._pnames
        args['state_names'] = self._snames
        for mapping in [self.istates,self.parameters]:
            for el in mapping:
                args[el] = mapping[el]
        args['ODE']=self._model
        args['t_steps'] = len(self.times)
        args['state_summations']=self._state_summations
        args['dataframe']=self.df
        return(args)

    def copy(self):
        return(ModelFramework(**self._arg_copy()))


    def MCMC(self,chain_inits=None,iterations=1000,cpu_cores=1,static_parameters=None,print_report=True):
        '''Launches Markov Chain Monte Carlo

        Parameters
        ----------
        chain_inits : list of dicts or dataframe
            list of dictionaries mapping parameters to their values or dataframe with parameter values as columns. Values
            will be used as the intial values for the Markov Chains, where the length of the list/dataframe implies the
            number of chains to start
        cpu_cores : int
            number of cores used in fitting, Default = 1
        print_report : bool
            Print a basic

        Returns
        -------
        pandas.DataFrame
            Data containing results from all markov chains

        '''
        #if a dataframe, lets pull out the values we need
        if isinstance(chain_inits,pd.DataFrame):
            chain_inits= [row.to_dict() for i,row in chain_inits[self.get_pnames()].iterrows()]

        #package SIn with parameters set and the iterations
        #jobs = [[SnI(self.df,**inits),iterations,int(iterations/2)] for inits in chain_inits]
        #nits=1000,burnin=None,static_parameters=None,print_progress=True
        jobs=[]
        MC_args={'nits':iterations,
                'static_parameters':static_parameters,
                'print_progress':False,
                'burnin':int(iterations/2)}
        if isinstance(chain_inits,int):
            args = self._arg_copy()
            for i in range(0,chain_inits):
                jobs.append([ModelFramework(random_seed=i,**args),MC_args])
        else:
            for inits in chain_inits:
                args = self._arg_copy()
                for el in inits:
                    args[el] = inits[el]#overwritting coppied elements
                jobs.append([ModelFramework(**args),MC_args])


        if cpu_cores == 1:
            posterior_list = []
            for job in jobs:
                posterior_list.append(job[0]._MarkovChain(**job[1]))
        else:
            posterior_list=self._parallelize(Chain_worker,jobs,cpu_cores)
        
        #annotated each posterior dataframe with a chain number
        for i in range(0,len(posterior_list)):
            posterior_list[i]['chain#']=i
        posterior = pd.concat(posterior_list)
        posterior.reset_index(drop=True,inplace=True)
        #setting medians
        p_median = {}
        for p in self.get_pnames():
            if p not in static_parameters:
                p_median[p] = np.exp(np.log(np.array(posterior[p].to_list()).mean(axis=0)))

        self.set_parameters(**p_median)
        #p_median= {}
        #report=["\nFitting Report\n==============="]
        #for col in list(self.get_pnames()):
        #    median,std = rawstats(posterior[col])
        #    report.append("parameter: {}\n\tmedian = {:0.3e}, Standard deviation = {:0.3e}".format(col,median,std))
        #    p_median[col]=median
        
        #self.set_parameters(**p_median) #reset params with new fits
        return(posterior)
        #if print_report:
        #    h,v = self.integrate(forshow=False)
        #    fs = self.get_fitstats(h,v)
        #    report.append("Median parameter fit stats:")
        #    report.append("\tChi = {:0.3e}\n\tAdjusted R-squared = {:0.3e}\n\tAIC = {:0.3e}".format(fs['Chi'],fs['AdjR^2'],fs['AIC']))
        #    print('\n'.join(report))

    def gradient(self,parameter_name,p_range,intialstates=None,seed_equilibrium=True,aggregate_enpoints=False,print_status=True):
        '''
        Iterativly launches numerical simulations with different Srs
        
        Parameters
        ----------
        Srs : array
            An array indicating the Sr of each simulation
        model : function
            Model used in numerical integration
        initvalues : list of tuples
            a list of tuples, where tuple[0]= member name and tuple[1]= inital value in simulation
        traits: dict of arrays
            A dictionary mapping trait names to arrays. Note that this must be compatible with the respective model
        t_final : int
            How long the simulation should run
        steps : int
            How many steps should be taken per t
        seed : bool, optional
            Seed the initial values of the next simulation with the results of the previous
        
        '''
        if isinstance(intialstates,type(None)):
            init = intialstates
        else:
            init = self.get_inits()
        num_sim = len(p_range)
        old_p = self.parameters[parameter_name]
        results = []
        if print_status:
            print("Preparing to run {} simulations between {} and {}".format(num_sim,min(p_range),max(p_range)))
        for i,p in enumerate(p_range):
            if print_status:
                print("{:.2f}% Complete".format(i/num_sim*100),end='\r')
            self.parameters[parameter_name] = p
            temp = self.integrate(inits=init,as_dataframe=True,sum_subpopulations=False)
            temp[parameter_name]=p
            if seed_equilibrium:
                last=np.array(temp.iloc[-1][self.get_snames(after_summation=False)])
                init = np.clip(last,a_min=1,a_max=None)#set init floor at 1
            if aggregate_enpoints:
                temp = temp.iloc[-1]            
            results.append(temp)
        if aggregate_enpoints:
            results = pd.DataFrame(results)
            results.drop('t',axis=1,inplace=True)
        if self._state_summations:#doing the summation last
            for sumpop in self._state_summations:
                results[sumpop] = results[self._state_summations[sumpop]].sum(axis=1)
                results.drop(self._state_summations[sumpop],inplace=True,axis=1)
        if print_status:
            print("100.00% Complete")
        self.parameters[parameter_name] = old_p
        return(results)

    #def find_endpoints(results):
    #    '''
    #    Build a dataframe composed of the last index from a list of dataframes
    #    '''
    #    m=[]
    #    for el in results:
    #        m.append(el.iloc[-1])
    #    return(pd.DataFrame(m))


