import pandas as pd
import numpy as np
import random as rd
from scipy.integrate import odeint
import multiprocessing
import matplotlib.pyplot as plt
from ODElib.Statistics import stats,Samplers

import warnings


def rawstats(pdseries):
    '''calculates raw median and standard deviation of posterior'''
    log_mean = np.log(pdseries).mean()
    median = np.exp(log_mean)
    log_std = np.log(pdseries).std()
    std = ((np.exp(log_std**2)-1)*np.exp(2*log_mean+log_std**2.0))**0.5
    return(median,std)

def _Chain_worker(model,argdict):
    '''Function called by pool for parallized fitting'''
    posterior = Samplers.MetropolisHastings(model,**argdict)
    return(posterior)

def _Equilibrium_worker(model,parameter_list=list()):
    '''
    Currently, this worker is not smart enough to determine if the system has reached equilibrium yet or not
    '''
    results = []
    for ps in parameter_list:
        #mod = model.integrate(parameters=model.get_parameters(**ps))
        mod = model.integrate(parameters=(ps,),as_dataframe=False,sum_subpopulations=False)
        eq = list(mod[-1,:])#storing final value
        eq.extend(ps)#
        results.append(eq)
    cols = model.get_snames()
    cols.extend(model.get_pnames())
    df = pd.DataFrame(results,columns=cols)
    return(df)


def _Fit_worker(model,parameter_list=list()):
    fits = []
    for ps in parameter_list:
        modcalc = model.integrate(parameters = (ps,),predict_obs=True,as_dataframe=False)
        chi = model.get_chi(modcalc)
        fits.append(list(ps)+[chi])
    df_fits = pd.DataFrame(fits,columns=model.get_pnames()+['chi'])
    return(df_fits)

class parameter:
    '''Parameter used in ModelFramework class

    The parameter class is used by the ModelFramework class to 
    initialize parameters. The goal of this class is to easily 
    maintain a parameter value, the underlying distribution of the parameter,
    and hyperparameters for defining the distribution. Moreover, this
    class is responsible for defining the random walks during MCMC fittings,
    therefore, any parameter-specifc random walks can be fully customized.

    Parameters
    ----------
    initials : float
        initial value of the parameter. If not specified, one is drawn. If a parameter
        needs to be an array, it must be specified here.
    stats_gen : scipy.stats.rv_continuous or scipy.stats.rv_discrete
        An instance of a scipy.stats.rv_continuous or scipy.stats.rv_discrete that can
        be call typical statistical functions (pdf/pmf, cdf, ppf, ect.).
    hyperparameters : dict
        A dictionary mapping hyperparameter names (indicated by the scipy distribution)
        to their values
    name : str
        the name of the parameter
    
    '''
    def __init__(self,stats_gen=None,hyperparameters=None,init_value=None,name=None):
        
        self.dist=stats_gen #scipy distribution
        self.hp=hyperparameters #store hyperparameters for shaping dist
        self.name=name
        if init_value:
            self.val = np.array(init_value) #store values as ndarray
        else:
            if not self.dist:
                raise ValueError("You must specify a scipy distribution if not passing a value")
            self.val=np.array((self.dist.rvs(**self.hp)))
        self._dim = self.val.shape #shape of val
    def fit(self,data):
        '''
        fits distribution to data and assigns hyperparameters
        '''
        shapeargs =self.dist.shapes.split(',')+['loc','scale']
        vals=self.dist.fit(data)
        for i,arg in shapeargs:
            self.hp[arg] = vals[i]


    def pdf(self,val=None):
        if self.dist:
            if val:
                return(self.dist.pdf(val,**self.hp))
            else:
                return(self.dist.pdf(self.val,**self.hp))
        else:
            return(1.0)
            
    def rwalk(self,std=.05):
        '''
        randomly walk the parameter
        '''
        stds=np.full(self._dim,std)
        #self.val = np.exp(np.log(self.val)+np.random.normal(0,stds))
        if self.dist:
            #logp=self.dist.logcdf(self.val,**self.hp)
            #here, we are taking the logcdf, adding a random var,
            #then exp to get ready for percent point function
            #pp = np.exp(logp+np.random.normal(0,stds))
            #self.val=self.dist.ppf(pp,**self.hp)
            self.val = np.exp(np.log(self.val)+np.random.normal(0,stds))
        else:
            #unconstrained random walk
            self.val = np.exp(np.log(self.val)+np.random.normal(0,stds))

    def has_distribution(self):
        '''
        Return true if the parameter has a distribution
        '''
        if self.dist:
            return(True)
        else:
            return(False)

    def __repr__(self):
        '''pretty printing'''
        outstr = [str(self.val)+'  ']
        if self.dist:
            outstr.append("(distribution:{}, ".format(self.dist.name))
            outstr.append("hyperparameters:{})".format(str(self.hp)))
        return(' '.join(outstr))

    def get_figure(self,samples=1000,logspace=False):
        '''
        returns a matplotlib Figure
        '''
        s = pd.Series(self.dist.rvs(size=samples,**self.hp))
        if logspace:
            axessubplot = s.hist(bins=np.logspace(np.log10(s.min()),np.log10(s.max()), 50))
            axessubplot.figure.gca().set_xscale("log")
        else:
            axessubplot = s.hist(bins=np.linspace(s.min(),s.max(), 50))
        axessubplot.set_title(self.name)

        return(axessubplot.figure)

    def __str__(self):
        return(self.__repr__())

    def copy(self):
        return(parameter(init_value=self.val,
                        stats_gen=self.dist,
                        hyperparameters=self.hp,
                        name=self.name)
                )


class ModelFramework():

    def __init__(self,ODE,parameter_names,state_names,dataframe=None,state_summations=None,
                t_end=5,t_steps=1000,random_seed=0,**kwargs):
        '''
        The ModelFramweork class acts to facilitate and expedite the analysis of different ODEs
        given some experimental data. Specifically, this class uses a Markov Chain Monte Carlo (MCMC) 
        implementation to fit and generate posterior distributions of those parameters.

        Parameters
        ----------
        ODE : function
            A callable ODE function with the arguments y, t, and ps. y is the argument for an array 
            of state variables that must match the function output. t is the argument for the time 
            array. ps is the argument for an array of parameters.
        parameter_names : list of str
            A list of strings should be specified containing the names of each parameter. The order
            of the parameters MUST match the unpacking order of the ps argument in the ODE function.
        state_names : list of str
            A list of strings should be specified containing the names of each state variable. The
            order of the state variable names MUST match the unpacking order of the y argument in 
            the ODE function.
        dataframe : pandas.DataFrame, optional
            A dataframe specifying the data for the model to be fit to. Dataframe columns must be
            specifed in one of two ways: 1) Dataframes contain the columns 'organism', 'time',
            'abundance', and 'variance' or 2) Dataframes contain the columns 'organism', 'time',
            'abundance', and 'replicate'. Option 1 assumes the user has appropriately calculated
            the variance and mean abundance at each timepoint for each species, while option 2 will
            automatically calculate the variance and means. organism names must corrospond to the
            state_names so fittings can be matched to the appropriate datapoints
        state_summations : dict, optional
            A dictionary mapping a representative name to the summation of ODE state variables.
        t_end : int
            Final timepoint of integration. If a dataframe is passed, the final timepoint will
            be set to the final timepoint in the dataframe.
        t_steps : int
            Number of timesteps to calculate during integration
        random_seed : int
            Random seed to be used by samplers
        '''
        
        #_pnames and _snames are essential for all other operations and should never change 
        self._pnames = tuple(parameter_names)
        self._snames = tuple(state_names)
        #_model references the ODE function. Unpacking order of states and parameters
        #should corrospond to the indicies of _snames and _pnames
        self._model = ODE
        
        #setting defaults for parameters and inital states
        self.parameters = {el:None for el in self._pnames}
        self.istates = {el:0 for el in self._snames} #initial state defaults are zero
        
        self.random_seed = random_seed#set the random seed for later

        #check to see if ODE state variables must be summed after each integration
        if state_summations:
            self._summations_index,self._summation_snames,self._sumkeep,self._suminds = self._get_summation_index(state_summations)
        else:
            self._summations_index,self._summation_snames,self._sumkeep,self._suminds = {},tuple(),tuple(),tuple()
        
        #data structures for doing fits
        self._obs_logabundance = {}
        self._obs_logsigma = {}
        self._obs_abundance = {}

        if isinstance(dataframe,pd.DataFrame):
            #we want to copy the dataframe as to not modify what was passed
            self.df = self._formatdf(dataframe.copy())
            self.times = np.linspace(0, max(self.df['time']),t_steps)#setup time
            self._samples=len(self.df)
            #building dicionaries used in fitting procdure
            self._pred_tindex,self._obs_logabundance,self._obs_logsigma=self._df_fitsetup()
        else:
            self.df=None
            self._samples=None
            self.times = np.linspace(0, t_end, t_steps)

        _is = {} #initial states as defined by the user
        _ps = {} #parameters as defined by the user        
        #the following identifies inital states and parameters in key word arguments (kwarg)
        if isinstance(self.df,pd.DataFrame):
            for org,abundance in self.df[self.df['time'] == 0]['abundance'].iteritems():
                if org not in _is:
                    _is[org] = abundance

        #we need to iterate over key word arguments to find intial state values and parameter values
        for el in kwargs:
            if el in self._pnames:
                _ps[el] = kwargs[el]
            if el in self._snames:
                _is[el] = kwargs[el] #overiding dataframe initial states        
        self.set_parameters(**_ps)#setting the parameters
        self.set_inits(**_is)#setting the intial states
        
        #store the number of parameters
        self._pnum=0
        for p in self.parameters:
            self._pnum += np.count_nonzero(self.parameters[p])
        

    def reset_dataframe(self,df):
        '''
        refreshes datastrucutres with new dataframe
        '''
        self.df = self._formatdf(df.copy())
        self.times = np.linspace(0, max(self.df['time']),len(self.times))#setup time
        self._pred_tindex,self._obs_logabundance,self._obs_logsigma=self._df_fitsetup()
        self._samples=len(self.df)
        _is = {}
        if isinstance(self.df,pd.DataFrame):
            for org,abundance in self.df[self.df['time'] == 0]['abundance'].iteritems():
                if org not in _is:
                    _is[org] = abundance
        self.set_inits(**_is)

    def _formatdf(self,df):
        '''
        This operation is aimed taking multilpe differently formated dataframes
        and returing a single object that is properly formatted
        '''
        df=df.sort_values(by=['organism','time'])
        if 'replicate' in df:#if the dataframe has replicates, calculate log mean and log sigma
            _df = df[['organism','time','abundance']]
            _df['log_abundance'] = np.log(_df['abundance'])
            dfagg = _df.groupby(by=['time','organism']).mean()
            dfagg['log_sigma'] = _df.groupby(by=['time','organism']).std()['log_abundance']
            dfagg=dfagg.reset_index(level='time')
            for sname in self._snames:
                if sname in dfagg.index:
                    self._obs_abundance[sname] = dfagg.loc[sname]['abundance'].to_numpy()
                    self._obs_logabundance[sname] = dfagg.loc[sname]['log_abundance'].to_numpy()
                    self._obs_logsigma[sname] = dfagg.loc[sname]['log_sigma'].to_numpy()
            df = dfagg
        else:
            df = df.set_index('organism')
            if 'abundance' in df and 'log_abundance' not in df:
                df['log_abundance'] = np.log(df['abundance'].to_numpy())
            if 'log_sigma' not in df:
                df['log_sigma'] = 1
                warnings.warn("log_sigma not found, setting log variance to 1")
            #should probably do something else here, like error checking
        return(df)

    def _df_fitsetup(self):
        '''
        defines datastrucutres needed for efficent fitting during MCMC
        '''
        _pred_tindex = {} #stores time index for predicted values
        for pred in set(self.df.index):
            if isinstance(self.df.loc[pred]['time'],pd.core.series.Series):
                _pred_tindex[pred] = np.r_[[np.where(abs(a-self.times) == min(abs(a-self.times)))[0][0] for a in self.df.loc[pred]['time']]]
            else:
                a = self.df.loc[pred]['time']
                _pred_tindex[pred] = np.r_[np.where(abs(a-self.times) == min(abs(a-self.times)))[0][0]]
        
        #we now have to store efficent datastrucutres that samplers can use
        #dictionaries have instant access!
        _obs_logabundance={}
        _obs_logsigma={}
        for sname in self.df.index:
            _obs_logabundance[sname] = self.df.loc[sname]['log_abundance'].to_numpy()
            _obs_logsigma[sname] = self.df.loc[sname]['log_sigma'].to_numpy()

        return(_pred_tindex,_obs_logabundance,_obs_logsigma)


    def _get_summation_index(self,summation_mapping):
        '''
        This funcitons maps which indices need to be summed after integration into
        a dictionary. For example, if there are state variables a,b and c in the ODE,
        being fit to data of a+b and c, then state variables a and b must be summed
        after each integration. passing a summation_mapping of {'ab':['a','b']} 
        allows for ab to be a summation of varaibles a and b in the ODE.
        
        Parameters
        ----------
        summation_mapping : dict
            name of representative state variable mapped to a list of state variables

        Returns
        -------
        tuple of index summation map and snames after summation
        '''
        sname_i = {sname:i for i,sname in enumerate(self._snames)}
        isum_summations = {}
        summed = set()
        i_newname={}
        for sumpop in summation_mapping:
            summation_indices=[] #keep track of all populations to sum together
            for pop in summation_mapping[sumpop]:
                if pop in summed:
                    raise ValueError("{} state varaiable cannot be used in two summations".format(pop))
                if pop not in self._snames:
                    raise ValueError("{} state varaiable is not a valid state name".format(pop))
                else:
                    summed.add(pop)
                    summation_indices.append(sname_i[pop])
            if len(summation_indices) < 1:#error incase users only specified one element for summation
                raise ValueError("Summation of {} has only {} specifed. Two or more are required for summations".format(sumpop,summation_mapping[sumpop][0]))
            #we will store the summation in the first index of the array after integration
            summation_indices.sort()
            isum=summation_indices[0]
            i_newname[isum] = sumpop #keep track of the new name for this index
            isum_summations[isum]=tuple(summation_indices)#summations of indecies are stored in first index
        summation_snames=[]
        summation_keep=[]
        #now we need redefine the names of the variables after the summation
        for i,sname in enumerate(self._snames):
            if i in i_newname:
                summation_snames.append(i_newname[i])
                summation_keep.append(i)
            elif sname not in summed:
                summation_snames.append(sname)
                summation_keep.append(i)
        #returning tuples because they should never be modified
        return(isum_summations,tuple(summation_snames),tuple(summation_keep),i_newname)

    def get_pnames(self):
        '''returns the names of the parameters used in the current model'''
        return(list(self._pnames))

    def get_snames(self,after_summation=True,predict_obs=False):
        '''returns the names of the state variables used in the current model
        
        Parameters
        ----------
        after_summation : bool
            Return the state variables as defined by the state_summations argument.
            If state_summations was not specified, state names of the ODE are returned.
            If False, the state names of the ODE are returned regardless of the definitons
            in state_summations
        predict_obs : bool
            Only return state names that have observations in the data
        '''
        if after_summation and self._summations_index:
            return(list(self._summation_snames))
        elif predict_obs:
            return(list(self._pred_tindex.keys()))
        else:
            return(list(self._snames))


    def __repr__(self):
        '''pretty printing'''
        outstr = ["Current Model = {}".format(str(self._model.__module__)+'.'+str(self._model.__name__)),
                "Parameters:",
        ]
        for p in self.get_pnames():
            outstr.append("\t{} = {}".format(p,self.parameters[p]))
        outstr.append("Initial States:")
        for s in self.get_snames(after_summation=False):
            outstr.append("\t{} = {}".format(s,self.istates[s]))
        if self._summations_index:
            outstr.append("Current State Summations")
            snames = self.get_snames(after_summation=False)
            summed_snames = self.get_snames(after_summation=True)
            for i in self._summations_index:
                summed='+'.join([snames[j] for j in self._summations_index[i]])
                outstr.append("\t{}={}".format(str(self._suminds[i]),summed))
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
                #if this parameter is being passed as ODElib.parameter, assign the whole datastructure
                if isinstance(kwargs[p],parameter):
                    self.parameters[p] = kwargs[p]
                    if not self.parameters[p].name:
                        self.parameters[p].name = p
                else:
                    #if this parameter has already been initialized, just change val and leave distribution
                    if self.parameters[p]:
                        self.parameters[p].val = kwargs[p]
                    #This parameter has not been assigned, make parameter datastructure
                    else:
                        self.parameters[p] = parameter(kwargs[p],None,None,p)
            else:
                raise Exception("{} is an unknown parameter. Acceptable parameters are: {}".format(p,', '.join(self._pnames)))

    def set_inits(self,**kwargs):
        '''set parameters for the model
        
        Parameters
        ----------
        **kwargs
            key word arguments, where keys are parameters and args are parameter values. Alternativly, pass **dict
        
        Raises
        ------
        ValueError
            ValueError is raised if a initial condition for a summation variable is passed and the summation of
            the ODE state variables are not equal
        '''
        s_set = set(self._snames) #sets are faster when checking membership!
        ss_set = set(self._summation_snames)
        checksums = {}#summations we need to check after inits have been set
        for s in kwargs:
            if s in s_set:
                self.istates[s] = kwargs[s]
            elif s in ss_set:
                checksums[s] = kwargs[s]
            else:
                raise Exception("{} is an unknown state variable. Acceptable parameters are: {}".format(s,', '.join(self._snames)))
        if checksums:
            #getting current intial values
            inits = self.get_inits()
            #map the sumation name to the correct inital value
            sname_i = {el:i for i,el in enumerate(self.get_snames(after_summation=True))}
            #for s in checksums:
            #    sumval=checksums[s]#what it should be
                #statesum is the current summation of the initial values
            #    statesum = sum(inits[tuple([self._summations_index[sname_i[s]]])])
            #    if sumval != statesum:
            #        sum_sname = np.array(self.get_snames(after_summation=False))
            #        err = ', '.join(sum_sname[[self._summations_index[sname_i[s]]]])
                    #msg="The initial value of {} is specified as {} in the dataframe, but summation of the intial ODE state variables '{}' is {}\nPlease specify correct inital values for ODE state variables"
                    #raise ValueError(msg.format(s,str(sumval),err,statesum))


    def get_inits(self,as_dict=False):
        '''
        returns the initial values used in integration

        Parameters
        ----------
        as_dict : bool
            If True, return inital states for ODE as a dictionary mapping state names
            to the initial values. Otherweise, return as an array according to
            the index of state_names
        '''
        if as_dict:
            return(self.istates)
        inits = np.array([self.istates[el] for el in self._snames])
        return(inits)
    
    #BROKEN!
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
        
        inits = self._lhs_samples(samples=10000,**kwargs) #inits is a dataframe
        
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
        '''
        return the ODE function used for integration
        '''
        return(self._model)

    def get_parameters(self,as_dict=False,**kwargs):
        '''return the parameters needed for integration
        
        Parameters
        ----------
        as_dict : bool, optional
            If true, return dict with parameter names mapped to values
        kwargs: optional
            pass a mapping of parameters to be packages for value return

        Return
        ------
        parameters
            numpy array of parameters ready for odeint or dict of parameters
        '''
        if as_dict:
            ps = {}
            for p in self.get_pnames():
                if p in kwargs:
                    ps[p] = kwargs[p]
                else:
                    ps[p] = self.parameters[p].val
        else:
            ps = []
            for p in self.get_pnames():
                if p in kwargs:
                    ps.append(kwargs[p])
                else:    
                    ps.append(self.parameters[p].val)
            ps = tuple([ps])
        return(ps)

    def get_numstatevar(self):
        '''returns the number of state varaibles'''
        return(len(self._snames))

    def _lhs_samples(self,samples=100,**kwargs):
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
        pdists = {}
        pstatic={}
        for p in self.parameters:
            #we can only do LHS samples on parameters that have distribuitons
            if p in kwargs:#allows samples without overwrite
                pdists[p] = kwargs[p]
            elif self.parameters[p].has_distribution():
                pdists[p] = self.parameters[p]
            else:
                pstatic[p]=self.parameters[p].val    
        df = Samplers.sample_lhs(parameter_dict=pdists,samples=samples)
        #add parameters that do not have distributions
        for p in pstatic:
            df[p]=pstatic[p]
        return(df)

    def get_residuals(self):
        mod = self.integrate(predict_obs=True)
        res = (mod.abundance - self.df.abundance)
        return(res)

    def integrate(self,inits=None,parameters=None,predict_obs=False,as_dataframe=True,sum_subpopulations=True):
        '''allows option to return model solutions at sample times

        Parameters
        ----------
        inits : numpy.array, optional
            ignore h0 and v0 in data and set the initial values for integration
        parameters : numpy.array, optional
            ignore stored parameters and use specified
        predict_obs : bool
            If True, only time points in df will be returned
        as_dataframe : bool
            If True, integration results are returned as a dataframe. This operation
            is expensive and should not be used when being called iterativly
        sum_subpopulations : bool
            apply state_summations after integration. If False, integration results
            are returned without summation. Default True

        Returns
        -------
        integration results as a dataframe or dictionary
            
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

        mod = odeint(func,y0=initials,t=self.times,args=ps)

        #subpopulation summations
        if sum_subpopulations and self._summations_index:
            for sumi in self._summations_index:
                #storing the summation in representative column
                mod[:,sumi]=mod[:,self._summations_index[sumi]].sum(axis=1)
            #we now need to keep summed colums and those not used in summations
            mod = mod[:,self._sumkeep]

        if as_dataframe:#this operation is expensive, avoid during iteration
            df = pd.DataFrame(mod)
            df.columns = self.get_snames(after_summation=sum_subpopulations)
            df['time'] = self.times
            if predict_obs:
                calc=pd.melt(df[self.get_snames(predict_obs=True)+['time']],id_vars=['time'])
                calc.columns = ['time','organism','abundance']
                calc=calc.set_index('organism')
                return(pd.concat([calc.loc[sname].iloc[self._pred_tindex[sname]] for sname in self.get_snames(predict_obs=True)]))                
            return(df)
        else:
            if predict_obs:
                mod_dict = {}
                for i,sname in enumerate(self.get_snames(after_summation=sum_subpopulations)):
                    if sname in self._pred_tindex:
                        mod_dict[sname]=mod[:,i][self._pred_tindex[sname]]#getting predicted values
                return(mod_dict)
            return mod

    def get_chi(self,mod_dict):
        '''goodness of fit test'''
        O=[]
        C=[]
        S=[]
        for sname in mod_dict:
            O.append(self._obs_logabundance[sname])
            C.append(np.log(mod_dict[sname]))
            S.append(self._obs_logsigma[sname])
        chi = stats.chi(O=np.concatenate(O,axis=0),
                        C=np.concatenate(C,axis=0),
                        S=np.concatenate(S,axis=0))
        return(chi)

    def get_Rsqrd(self,mod_dict):
        abundance_dict = {el:np.exp(self._obs_logabundance[el]) for el in self._obs_logabundance}
        Rsqrd=stats.Rsqrd(C_dict=mod_dict,O_dict=abundance_dict)
        return(Rsqrd)

    def get_AIC(self,chi):
        AIC=stats.AIC(chi,self._pnum)
        return(AIC)

    def get_adjRsqrd(self,mod_dict,Rsqrd=None):
        if not Rsqrd:
            Rsqrd = self.get_Rsqrd(mod_dict)
        adjRsqrd = stats.get_adjusted_rsquared(Rsqrd,self._samples,self._pnum)
        return(adjRsqrd)

    def get_fitstats(self,prediction_dict=dict()):
        '''return dictionary of adjusted R-squared, Chi, and AIC of current parameters'''
        fs = {}
        if not prediction_dict:
            prediction_dict = self.integrate(predict_obs=True,as_dataframe=False)
        fs['Chi'] = self.get_chi(prediction_dict)
        fs['R^2'] = self.get_Rsqrd(prediction_dict)
        fs['AIC'] = self.get_AIC(fs['Chi'])
        return(fs)
   
    # set optimal parameters 
    def set_best_params(self,posteriors):
        im = posteriors.loc[posteriors.chi==min(posteriors.chi)].index[0]
        bestchain = posteriors.iloc[im]["chain#"]
        posteriors = posteriors[posteriors["chain#"]==bestchain]
        self.set_parameters(**posteriors.loc[im][self.get_pnames()].to_dict())
        self.set_inits(**{o:posteriors.loc[im][self.get_pnames()].to_dict()[o+'0'] for o in self._snames})

    #function for plotting uncertainty once model has been run 
    def plot_uncertainty(self,ax,posteriors,variable,ntimes=100):
        for a in range(ntimes):
            im = rd.choice(posteriors.index)
            self.set_inits(**{o:posteriors.loc[im][self.get_pnames()].to_dict()[o+'0'] for o in self._snames})
            self.set_parameters(**posteriors.loc[im][self.get_pnames()].to_dict())
            mod = self.integrate()
            ax.plot(mod.time,mod[variable],c=str(0.8),lw=1,zorder=1)

    #FIX ME?
    def _rand_walk(self,pdict):
        _pdict={}
        stds=.05
        #stds=.1
        for p in pdict:
            if isinstance(pdict[p],np.ndarray):
                _pdict[p]=np.exp(np.log(pdict[p]) + np.random.normal(0, stds,pdict[p].shape))
            else:
                _pdict[p]=np.exp(np.log(pdict[p]) + np.random.normal(0, stds))
        return(_pdict) 

    
    def _parallelize(self,func,args,cores,print_stmt=True):
        '''Wrapper for Parallelization of jobs
        
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
            list of outputs from funcself
        
        '''
        if cores > multiprocessing.cpu_count():
            Warning("More cores specified than avalible, cpu_cores set to maximum avalible\n")
            cores=multiprocessing.cpu_count()
        if print_stmt:
            print("Starting {} processes with {} cores".format(len(args),cores),end='\r')
        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.starmap(func,args)
        pool.join()
        pool.close()
        if print_stmt:
            print("Starting {} processes with {} cores\t[DONE]".format(len(args),cores))
        return(results)

    def _package_parameters(self,num_workers,parameter_dataframe):
        '''Takes a parameter dataframe and distributes among workerlist '''
        #ensure all parameters are in the dataframe
        parameter_names = self.get_pnames()
        for p in parameter_names:
            if p not in parameter_dataframe:
                parameter_dataframe[p] = [self.parameters[p]]*len(parameter_dataframe)
        parameter_dataframe=parameter_dataframe[parameter_names]#ensure order is correct
        worklist=[list() for i in range(0,num_workers)]
        for i,row in parameter_dataframe.iterrows():#distributing jobs among workers
            worklist[i%num_workers].append(tuple(row))
        return(worklist)

    def fit_survey(self,samples=1000,cpu_cores=1):
        '''
        samples prior distribuiton with LHS scheme for fits
        '''
        ps = self._lhs_samples(samples)
        worklist=self._package_parameters(cpu_cores,ps)
        jobs=[]
        while worklist:
            jobs.append([self.copy(),worklist.pop()])
        if cpu_cores ==1:
            results = []
            for job in jobs:
                results.append(_Fit_worker(job[0],job[1]))
        else:                
            results = self._parallelize(_Fit_worker,jobs,cores=cpu_cores)
        results = pd.concat(results)
        return(results)


    def explore_equilibriums(self,samples=1000,cpu_cores=1,**parameter_mapping):
        '''Launch 

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
        ps = self._lhs_samples(samples,**parameter_mapping)
        worklist=self._package_parameters(cpu_cores,ps)
        jobs=[]
        while worklist:
            jobs.append([self.copy(),worklist.pop()])
        if cpu_cores ==1:
            results = []
            for job in jobs:
                results.append(_Equilibrium_worker(job[0],job[1]))
        else:                
            results = self._parallelize(_Equilibrium_worker,jobs,cores=cpu_cores)
        results = pd.concat(results)
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
        ps = self._lhs_samples(kwargs,samples)
        worklist = self._package_parameters(cpu_cores,ps)
        jobs=[]
        while worklist:
            jobs.append([self.copy(),worklist.pop()])
        if cpu_cores ==1:
            results = []
            for job in jobs:
                results.append(_Fit_worker(job[0],job[1]))
        else:                
            results = self._parallelize(_Fit_worker,jobs,cores=cpu_cores)
        fits=[]
        for workerfit in results:
            fits.extend(workerfit)
        fitdf=pd.DataFrame(fits,columns = self.get_pnames()+['chi'])
        return(fitdf)
        



    def copy(self,overwrite=dict()):
        '''
        Creates a copy of the current ModelFramework. All attributes are copied automatically. Typically used for
        creating instances for parallel operations.

        Parameters
        ---------
        overwrite : dict
            A dictionary containing new initial states or parameters

        Returns
        -------
        ModelFramework
        '''
        #make a new instance of framework
        newmod = ModelFramework(ODE=self.get_model(),
                                parameter_names=self._pnames,
                                state_names=self._snames,
                                    )
        #we need specifically iterate though parameters to make sure those objects are not being shared between model instances
        for p in self.parameters:
            newmod.parameters[p] = self.parameters[p].copy()
        already_copied = set(['_model','_pnames','_snames','parameters'])
        for attr in self.__dict__:
            if attr not in already_copied:
                if isinstance(self.__dict__[attr],(list,dict,pd.DataFrame,np.ndarray)):
                    newmod.__dict__[attr]=self.__dict__[attr].copy() #deep copy specific datastrucutres to make sure we are not passing by reference
                else:
                    newmod.__dict__[attr]=self.__dict__[attr]

        #we now must overwrite any parameters or initial states
        _ps={}
        _is={}
        for el in overwrite:
            if el in newmod._pnames:
                _ps[el] = overwrite[el] #overriding
            if el in newmod._snames:
                _is[el] = overwrite[el] #overriding dataframe initial states
        if _ps:
            newmod.set_parameters(**_ps)
        if _is:            
            newmod.set_inits(**_is)
        return(newmod)


    def MCMC(self,chain_inits=1,iterations_per_chain=1000,cpu_cores=1,static_parameters=list(),print_report=True,fitsurvey_samples=1000,sd_fitdistance=3.0):
        '''Launches Markov Chain Monte Carlo

        A Markov Chain Monte Carlo fitting protocol is used to find best fits. Note that chains can only be computed
        by a single CPU, therefore, increasing the number of cpu_cores for a single chain with many iterations will
        not improve performance.

        Parameters
        ----------
        chain_inits : list of dicts or dataframe
            list of dictionaries mapping parameters to their values or dataframe with parameter values as columns. Values
            will be used as the intial values for the Markov Chains, where the length of the list/dataframe implies the
            number of chains to start
        iterations_per_chain : int
            number of iterations to perform during MCMC chain. Default = 1000
        cpu_cores : int
            number of cores used in fitting, Default = 1
        print_report : bool
            Print a basic
        static_parameters : list, optional
            A list of parameters that do not change during MCMC fitting
        fitsurvey_samples : int
            The number of samples to take from multidimensional to search for good inital fits. Default = 1000
        sd_fitdistance : float
            The number of standard deviations away from data that is acceptable as an initial fit. Default = 3.0

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
        
        MC_args={'nits':iterations_per_chain,
                'static_parameters':static_parameters,
                'print_progress':False,
                'burnin':int(iterations_per_chain/2)}
        #if passing a integer, user is not specifying inits ps for chains. We will pick by finding startpoints
        #where integration does not fail by calling fit_survey
        if isinstance(chain_inits,int):
            #quickly explore parameter space using LHS
            fitsurvey = self.fit_survey(cpu_cores=cpu_cores,samples=fitsurvey_samples)
            #removing parameter draws that cause integration failures
            fitsurvey.dropna(inplace=True)
            if fitsurvey.empty:#fit survey is empty (due to no ps to sample or VERY bad priors)
                #create a dummy datastrucutre so arguments are not overwritten
                initps=pd.DataFrame([[]]*chain_inits)
                warnings.warn("Pre-sampling of Multidimentional space failed")
            else:
                #calculating threshold based on standard deviations away
                calc = {sname:np.exp(self._obs_logabundance[sname]+sd_fitdistance*self._obs_logsigma[sname]) for sname in self._obs_logabundance}
                cutchi = self.get_chi(calc)
                if sum(fitsurvey['chi']<cutchi) == 0:
                    raise ValueError("Preliminary sampling found no parameter sets which meet the minimal threshold \n \
                        Try: \n \
                             1. Increasing sd_fitdistance \n \
                             2. Increasing fitsurvey_samples \n \
                             3. Different priors and / or different parameter guesses")
                initps = fitsurvey[fitsurvey['chi']<cutchi].sample(chain_inits,replace=True) #only sample acceptable fits
            for i in range(0,chain_inits):
                newmodel = self.copy(overwrite=initps.iloc[i].to_dict())#get a copy of arguments with parameters overwritten
                newmodel.random_seed=i
                jobs.append([newmodel,MC_args])#setting a new random seed per chain
        else:
            for i,inits in enumerate(chain_inits):
                newmodel = self.copy(overwrite=inits)
                newmodel.random_seed=i
                jobs.append([newmodel,MC_args])

        if cpu_cores == 1:
            posterior_list = []
            for job in jobs:
                posterior_list.append(Samplers.MetropolisHastings(job[0],**job[1]))
        else:
            posterior_list=self._parallelize(_Chain_worker,jobs,cpu_cores)
        
        #annotated each posterior dataframe with a chain number
        for i in range(0,len(posterior_list)):
            posterior_list[i]['chain#']=i
        posterior = pd.concat(posterior_list)
        posterior.reset_index(drop=True,inplace=True)
        #setting medians
        #p_median = {}
        #for p in self.get_pnames():
        #    if p not in static_parameters:
        #        p_median[p] = np.exp(np.log(np.array(posterior[p].to_list()).mean(axis=0)))
        #print("Setting parameters to median of posterior")
        #self.set_parameters(**p_median)
        #return(posterior)
        if print_report:
            p_median= {}
            report=["\nFitting Report\n==============="]
            for col in list(self.get_pnames()):
                median,std = rawstats(posterior[col])
                if (median !=0.0) and (std !=0.0):
                    report.append("parameter: {}\n\tmedian = {:0.3e}, Standard deviation = {:0.3e}".format(col,median,std))
                    p_median[col]=median
            self.set_best_params(posterior)
            mod = self.integrate(predict_obs=True,as_dataframe=False)
            fs = self.get_fitstats(mod)
            report.append("\nMedian parameter fit stats:")
            report.append("\tChi = {:0.3e}\n\tR-squared = {:0.3e}\n\tAIC = {:0.3e}".format(fs['Chi'],fs['R^2'],fs['AIC']))
            print('\n'.join(report))
        return(posterior)

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
            #temp is an numpy array
            temp = self.integrate(inits=init,as_dataframe=False,sum_subpopulations=False)
            #temp[parameter_name]=p
            if seed_equilibrium:
                last=temp[-1,:]
                init = np.clip(last,a_min=.001,a_max=None)#set init floor at 1
            if aggregate_enpoints:
                temp=temp[-1,:]#get last element
                result = np.zeros(temp.shape[0]+1)#make new array
                result[:-1]=temp#fill array with final states
                result[-1] = p #add parameter value in last array
            else:
                shape = list(temp.shape) #get the array shape
                shape[-1] += 1 #grow by one column at the end
                result = np.zeros((shape)) #create new array
                result[:,:-1] = temp #fill array with previous data
                result[:,-1] = p #fill final column with parameter value
            results.append(result)
        #if aggregate_enpoints:
            #results = pd.DataFrame(results)
            #results.drop('t',axis=1,inplace=True)
        #if self._state_summations:#doing the summation last
        #    for sumpop in self._state_summations:
        #        results[sumpop] = results[self._state_summations[sumpop]].sum(axis=1)
        #        results.drop(self._state_summations[sumpop],inplace=True,axis=1)
        if print_status:
            print("100.00% Complete")
        self.parameters[parameter_name] = old_p
        col = self.get_snames()
        col.append(parameter_name)
        results = pd.DataFrame(results,columns=col)
        return(results)

    #def find_endpoints(results):
    #    '''
    #    Build a dataframe composed of the last index from a list of dataframes
    #    '''
    #    m=[]   
    #    for el in results:
    #        m.append(el.iloc[-1])
    #    return(pd.DataFrame(m))

    def _calc_stds(self,state):
        logabundance = self._obs_logabundance[state]
        logstd = self._obs_logsigma[state]
        low = np.exp(logabundance) - np.exp(logabundance-logstd) 
        high = np.exp(logabundance+logstd) - np.exp(logabundance)
        return(np.array([low,high]))

    def plot(self,states=None,overlay=dict()):
        if not states:
            states = self.get_snames(predict_obs=True)
        rplt = (len(states)%2+len(states)) /2
        f,ax = plt.subplots(int(rplt),2,figsize=[9,4.5])
        mod = self.integrate()
        for i,state in enumerate(states):
            if state in self.df.index:
                ax[i].errorbar(self.df.loc[state]['time'],
                            self.df.loc[state]['abundance'],
                            yerr=self._calc_stds(state)
                            )
            ax[i].set_xlabel('Time')
            ax[i].set_ylabel(state+' ml$^{-1}$')
            ax[i].semilogy()
            if state in mod:
                ax[i].plot(self.times,mod[state])
                if state in overlay:
                    for el in overlay[state]:
                        ax[i].plot(self.times,mod[el])
        return(f,ax)

