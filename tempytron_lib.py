#pkgs/utilities
import numpy as np
import pandas as pd
from functools import partial
from copy import deepcopy
import time
from multiprocessing import Pool

#############################X pattern generation
def gen_spk_data(n, fr=5*1/1000., T=1000.):
    return gen_event_seq_data(np.repeat(fr*T,n), T, 'spk_idx','spk_times')
 
def gen_feature_occurence_data(cf_mean, T):
    return gen_event_seq_data(cf_mean, T, 'occur_idx','occur_times')
 
def gen_event_seq_data(ev_count_means, T, ev_type_label,ev_time_label):
    counts = np.random.poisson(lam=ev_count_means)
    data=pd.DataFrame({ev_type_label:np.repeat(range(len(ev_count_means)),counts),\
                       ev_time_label:np.random.random(np.sum(counts))*T          })
    data[ev_type_label]=data[ev_type_label].astype(int)
    return data.sort_values(by=ev_time_label).reset_index(drop=True) 

def get_input_pattern(feature_data,fea_count_means,num_syn,bkgd_activity_duration=1000.,bkgd_avg_syn_firingrate=5*1/1000.):
    feature_duration=feature_data['ensemble_paras']['feature_duration']
    fea_patterns=feature_data['data']
    bkgd_df=gen_spk_data(num_syn, bkgd_avg_syn_firingrate, bkgd_activity_duration)
    
    #initialize pattern as background pattern
    pattern_df=bkgd_df.copy()
    
    occur_df=gen_feature_occurence_data(fea_count_means, bkgd_activity_duration)
    if not occur_df.empty:  
        #shift spikes and append occurences
        fac=(pattern_df.spk_times[:,np.newaxis]>occur_df.occur_times[np.newaxis,:]).sum(axis=1)
        pattern_df.spk_times+=fac*feature_duration
        
        features=pd.concat([fea_patterns[occur_idx] for occur_idx in occur_df.occur_idx.values],ignore_index=True)
        
        fea_length=np.array([len(fea_pattern) for fea_pattern in fea_patterns])
        occur_df.occur_times+=feature_duration*occur_df.index.values
        features.spk_times+=np.repeat(occur_df.occur_times.values,fea_length[occur_df.occur_idx.values])

        pattern_df=pattern_df.append(features,ignore_index=True).sort_values(by='spk_times').reset_index(drop=True)
        
    return pattern_df, occur_df

def make_feature_data(seed,neu_paras,num_fea,fea_count_means,fea_labels):
    ens_para_dict={'feature_duration':50.,'avg_syn_firingrate':5*(1/1000.)} #convert to Hz to /ms
    seed += 1
    np.random.seed(seed) #reserve integer-valued seeds for pattern generation
    fea_patterns=[gen_spk_data(neu_paras['num_syn'], ens_para_dict['avg_syn_firingrate'], ens_para_dict['feature_duration'])\
                                                                   for fea in range(num_fea)] #as a list of dataframes
    features={'data':fea_patterns,'ensemble_paras':ens_para_dict}
    return (features,fea_count_means, fea_labels),seed
  
############################X neuron model  
def gen_neuron_paras(tau_syn=5.,tau_mem=20.,v_thresh=1.,num_syn=500): #parameter dictionary passed to functions
    eta=tau_mem/tau_syn
    v_norm=eta**(eta/(eta-1))/(eta-1)
    neu_paras={
        'tau_syn':tau_syn,
        'tau_mem':tau_mem,
        'v_norm':v_norm,
        'v_thresh':v_thresh,
        'num_syn':num_syn
        }
    return neu_paras

########################X model output  
def get_vdv_rootfunc(time,logD,logDs,sgnD,sgnDs,neuron_paras):
    expm=sgnD *np.exp(logD -time/neuron_paras['tau_mem'])
    exps=sgnDs*np.exp(logDs-time/neuron_paras['tau_syn'])
    v =    neuron_paras['v_norm']*( expm                         - exps)
    dvdt = neuron_paras['v_norm']*(-expm/neuron_paras['tau_mem'] + exps/neuron_paras['tau_syn'])
    return v - neuron_paras['v_thresh'], dvdt
 
def root_finding(tlower,tupper,root_func):
    assert tlower<tupper, 'bisection bounds not ordered!'
    tol = 1e-8 ; f=2*tol
    it = 0; max_iter = 200
    told = 0; tnew = tlower 
    bisectFlag = False
    while ( ( (np.fabs(tnew - told)> tol) and (tupper-tlower>tol) ) and (it <= max_iter) and (np.fabs(f)> tol) ):
        it+=1
        told = deepcopy(tnew)
        f, df = root_func(told)
        tlower = tlower if (f>=0) else deepcopy(told)
        tupper = deepcopy(told) if (f>=0) else tupper
        assert tlower<tupper, 'bisection bounds not ordered!'
        if (it > 30):
            bisectFlag = True
            print("passed 30 iterations. Switch to bisection method")
        if (bisectFlag == False):
            tnew = told - f/df
            if (tnew < tlower) or (tnew > tupper):
                print("root estimate out of bounds. Do a bisect, then return to newton")
                tnew = (tupper + tlower)/2
        else:
            tnew = (tupper + tlower)/2
    if it>max_iter:
        print('max iterations reached!')
    return tnew
  

def precompute_data_for_iteration(neuron_paras,pattern_df,weights):
    w_seq=weights[pattern_df.spk_idx]
    logw_seq=np.log(np.fabs(w_seq))
    Dvec= np.cumsum(np.sign(w_seq)*np.exp(logw_seq+pattern_df.spk_times.values/neuron_paras['tau_mem']))
    Dsvec=np.cumsum(np.sign(w_seq)*np.exp(logw_seq+pattern_df.spk_times.values/neuron_paras['tau_syn']))
    logDvec=np.log(np.fabs(Dvec))
    logDsvec=np.log(np.fabs(Dsvec))
    sgnDvec=np.sign(Dvec)
    sgnDsvec=np.sign(Dsvec)
    return logDvec,logDsvec,sgnDvec,sgnDsvec

#@profile  
def get_outspk_times(pattern_df,neu_paras,logDvec,logDsvec,sgnDvec,sgnDsvec,neuron_model_is='sst',get_threshcrossing_Max_only=False,output_voltage=False):
    outspk_times=list()
    tau_mem=neu_paras['tau_mem']
    tau_syn=neu_paras['tau_syn']
    v_norm=neu_paras['v_norm']
    v_thresh=neu_paras['v_thresh']
    eta=tau_mem/tau_syn
    logeta=np.log(eta)
    taufrac=tau_mem/(1-eta)
    inspk_times=np.append(pattern_df.spk_times.values,np.Inf)                                   #add input spike at infinity to have complete and consistent interval looping
    inspk_times_tau_mem=inspk_times/tau_mem
    inspk_times_tau_syn=inspk_times/tau_syn
    logDvec_mod=deepcopy(logDvec)                                                               #modified versions contain effects of any output spikes. Used to get voltage values at input spike times
    sgnDvec_mod=deepcopy(sgnDvec)
    #init sst paras
    vMax=-np.Inf                                                                                #n.b. actually only need if using margin (not implemented yet)
    tMax=None
    jMax=None
    for j,inspk_time in enumerate(inspk_times[:-1]):                                            #-1: don't include appended inf spiketime
        if sgnDvec_mod[j]>0 and sgnDsvec[j]>0:                                                  #maximum exists? (for case when future input spikes are shunted) n.b. next 3 lines are the program's computational bottleneck  
            tMax_shunted=(logDvec_mod[j]-logDsvec[j]-logeta)*taufrac
            is_interior_maximum=(tMax_shunted>inspk_time and tMax_shunted<inspk_times[j+1])
            vcheck=np.exp(logDsvec[j]/(1-eta)+logDvec_mod[j]/(1-1/eta)) if is_interior_maximum \
                          else v_norm*(np.exp(logDvec_mod[j]-inspk_times_tau_mem[j+1])-np.exp(logDsvec[j]-inspk_times_tau_syn[j+1]))
            if vcheck>v_thresh:    
                if get_threshcrossing_Max_only:
                    jMax=j
                    tMax=tMax_shunted
                    vMax=vcheck if is_interior_maximum else np.exp(logDsvec[j]/(1-eta)+logDvec_mod[j]/(1-1/eta))
                    outspk_times.append(None)#outspike exists but not computed
                    break
                tupper=tMax_shunted if is_interior_maximum else inspk_times[j+1]
                Dvec_mod_tmp=sgnDvec_mod[j]*np.exp(logDvec_mod[j])
                sgnDvec_mod_tmp=np.sign(Dvec_mod_tmp)
                logDvec_mod_tmp=np.log(np.fabs(Dvec_mod_tmp))
                vdv_rootfunc=partial(get_vdv_rootfunc,logD=logDvec_mod_tmp,logDs=logDsvec[j],sgnD=sgnDvec_mod_tmp, \
                                                                                sgnDs=sgnDsvec[j],neuron_paras=neu_paras)
                spike_term=0
                tlower=deepcopy(inspk_time)
                while vcheck>v_thresh: # or interior_maximum_cond                              #run root finding iteratively until voltage at next spike is subthreshold and no interior maximum
                    outspk_time=root_finding(tlower,tupper,vdv_rootfunc)
                    outspk_times.append(outspk_time)
                    spike_term+=-v_thresh/v_norm*np.exp(outspk_time/tau_mem)
                    Dvec_mod_tmp=sgnDvec_mod[j]*np.exp(logDvec_mod[j])+spike_term
                    sgnDvec_mod_tmp=np.sign(Dvec_mod_tmp)
                    logDvec_mod_tmp=np.log(np.fabs(Dvec_mod_tmp))
                    vdv_rootfunc=partial(get_vdv_rootfunc,logD=logDvec_mod_tmp,logDs=logDsvec[j],sgnD=sgnDvec_mod_tmp,\
                                                            sgnDs=sgnDsvec[j],neuron_paras=neu_paras)
                    vcheck=vdv_rootfunc(tupper)[0]+v_thresh
                    if neuron_model_is=='sst':
                        break
                    tlower=outspk_time
                
                #update total interval contribution
                Dvec_mod_tmp=sgnDvec_mod[j+1:]*np.exp(logDvec_mod[j+1:])+spike_term            #doesn't handle j=end case since inspike_times[j+1]=np.Inf and precompute only up to j
                sgnDvec_mod[j+1:]=np.sign(Dvec_mod_tmp)                                        #so add them explicitly here
                logDvec_mod[j+1:]=np.log(np.fabs(Dvec_mod_tmp))
                
                if neuron_model_is=='sst':
                    break
            else:
                if neuron_model_is=='sst':
                    if vcheck>vMax:
                        vMax=deepcopy(vcheck)
                        jMax=j
                        tMax=tMax_shunted if is_interior_maximum else inspk_times[j+1]
        
    if output_voltage:  
        voltage_at_spikes = v_norm*(sgnDvec_mod*np.exp(logDvec_mod-inspk_times_tau_mem[:-1]) \
                                      - sgnDsvec*np.exp(logDsvec   -inspk_times_tau_syn[:-1]))
    if tMax is None: #if voltage never has maximum (usually implies pathological weight trajectory), assign values based on last input spike
        jMax=j
        tMax=inspk_time+(1/tau_syn-1/tau_mem)*np.log(tau_mem/tau_syn) #time to maximum of unit weight input, could also try using maximum: (logDvec_mod[j]-logDsvec[j]-logeta)*taufrac
        vMax=v_norm*(sgnDvec_mod[jMax]*np.exp(logDvec_mod[jMax]-inspk_times_tau_mem[:-1][jMax]) \
                                      - sgnDsvec[jMax]*np.exp(logDsvec[jMax]   -inspk_times_tau_syn[:-1][jMax]))
        if vMax>v_thresh:
             print('Maximum attained!!!!')
    
    outvars=outspk_times,vMax,tMax,jMax#,voltage_at_spikes
    #if output_voltage:
       #outvars=tuple(list(outvars).append(voltage_at_spikes))
    return outvars
  
############0 learn
#@profile  
def get_corr_inspk_eligibilities(neuron_paras,inspk_times,outspk_times,logDvec,logDsvec,sgnDvec,sgnDsvec):
    tau_mem=neuron_paras['tau_mem']
    tau_syn=neuron_paras['tau_syn']
    inspk_times=np.append(inspk_times,np.Inf)

    tau_mean=(tau_mem*tau_syn)/(tau_mem+tau_syn)
    taum_fac=         sgnDvec*tau_mem/2*( np.exp( logDvec-2*inspk_times[:-1]/tau_mem ) - np.exp( logDvec-2*inspk_times[1:]/tau_mem )) \
                    +sgnDsvec*tau_mean *(-np.exp(logDsvec-  inspk_times[:-1]/tau_mean) + np.exp(logDsvec-  inspk_times[1:]/tau_mean)) 
    taus_fac=        sgnDsvec*tau_syn/2*( np.exp(logDsvec-2*inspk_times[:-1]/tau_syn ) - np.exp(logDsvec-2*inspk_times[1:]/tau_syn )) \
                    + sgnDvec*tau_mean *(-np.exp( logDvec-  inspk_times[:-1]/tau_mean) + np.exp( logDvec-  inspk_times[1:]/tau_mean)) 
    taum_sum=np.cumsum(taum_fac[::-1])[::-1]
    taus_sum=np.cumsum(taus_fac[::-1])[::-1]
    
    kernel_corr=(neuron_paras['v_norm']**2)*(np.sign(taum_sum)*np.exp(inspk_times[:-1]/tau_mem+np.log(np.fabs(taum_sum))) \
                                           + np.sign(taus_sum)*np.exp(inspk_times[:-1]/tau_syn+np.log(np.fabs(taus_sum))))       
    if len(outspk_times)==0:  
        return kernel_corr
    else:
        assert outspk_times[0] is not None,"must compute outspike time!"
        outspk_times=np.asarray(outspk_times)
        out_spk_after=(inspk_times[:-1,np.newaxis]<outspk_times[np.newaxis,:])
        exp_out_mem=np.exp((inspk_times[:-1,np.newaxis]-outspk_times[np.newaxis,:])/tau_mem)
        exp_out_syn=np.exp((inspk_times[:-1,np.newaxis]-outspk_times[np.newaxis,:])/tau_syn)
        
        reset_corr=-neuron_paras['v_norm']*neuron_paras['v_thresh']*tau_mem/2* \
              np.sum( ((tau_mem-tau_syn)/(tau_mem+tau_syn))*((~out_spk_after)/exp_out_mem) \
                    + (exp_out_mem-(2*tau_syn/(tau_mem+tau_syn))*exp_out_syn)*out_spk_after \
                    ,axis=1)    
        return kernel_corr + reset_corr  
#@profile  
def get_neuron_eligibilities(pattern_df,neu_paras,normalize=False,num_outspks=None,current_weights=None):
    neuron_eligibilities=pattern_df.groupby('spk_idx').elig.sum()
    if normalize:
        syn_int =pattern_df.spk_idx.value_counts()*neu_paras['v_norm']*(neu_paras['tau_mem']-neu_paras['tau_syn'])
        volt_int=np.abs( (neu_paras['v_norm']  *(neu_paras['tau_mem']-neu_paras['tau_syn']))*current_weights[pattern_df.spk_idx].sum() \
                        -neu_paras['v_thresh']* neu_paras['tau_mem']                       *num_outspks)
        neuron_eligibilities/=(syn_int*volt_int)
    return neuron_eligibilities
    
def get_weight_change(current_weights,error_sign,learning_rate,neuron_eligibilities,learning_rule_is,top_elig_fraction=0.1,plasticity_induction_threshold=1e-3): 
    if learning_rule_is=='corr_top_p': 
        eligible_subset=neuron_eligibilities.sort_values(ascending=False).index.values[:int(top_elig_fraction*len(current_weights))]
        eligibility_magnitude=1.
    elif learning_rule_is=='corr_thresh':
        tmp=np.zeros(current_weights.shape)
        tmp[neuron_eligibilities.index.values]=neuron_eligibilities.values
        eligible_subset=(tmp>plasticity_induction_threshold) #boolean index
        #eligible_subset=neuron_eligibilities.index.values
        #eligibility_magnitude=neuron_eligibilities.values
        eligibility_magnitude=1.
    else: #apply to all
        eligible_subset=neuron_eligibilities.index.values
        eligibility_magnitude=neuron_eligibilities
    
    weight_change=np.zeros(current_weights.shape)
    weight_change[eligible_subset]=learning_rate*error_sign*eligibility_magnitude
    
    if learning_rule_is=='corr_thresh' and error_sign>0: #gutig heuristic to improve threshold-based correlation learning for sst near capacity
        epsilon=1e-2
        weight_change[~eligible_subset]=epsilon*learning_rate*error_sign*eligibility_magnitude #index must be boolean
    
    return weight_change  
  
def momentum_step(current_weight_change,prev_weight_change,momentum_factor=0.99):
    #nonzero_weightchange=np.ones(len(current_weight_change),dtype=bool)   #adds momentum to all weights (sst)
    nonzero_weightchange=(current_weight_change!=0)                        #only adds momentum to finite weight changes (mst)
    current_weight_change+=momentum_factor*prev_weight_change  
    prev_weight_change[nonzero_weightchange]=current_weight_change[nonzero_weightchange] #only update previous weight change for weights that have a finite pre-momentum weight change 
    return current_weight_change*nonzero_weightchange,prev_weight_change                 #only update wieghts that have a finite pre-momentum weight change
    #return current_weight_change,prev_weight_change                 #only update wieghts that have a finite pre-momentum weight change

def warmup(current_weights,trials_per_cycle,n_cycles,neu_paras,seed,train_specs,get_threshcrossing_Max_only):
    num_syn=neu_paras['num_syn']
    seed+=1
    np.random.seed(seed)
    if train_specs['neuron_model_is']=='sst':
        target_rate=1
        warmup_learning_rate=1e-5
        pattern_activity_duration=500

    elif train_specs['neuron_model_is']=='mst':
        target_rate=5.
        warmup_learning_rate=1e-3
        pattern_activity_duration=1000
    cycle = 0
    it=0
    while cycle < n_cycles: 
        trial = cycle*trials_per_cycle
        store_num_spks=[]
        while it < trial+trials_per_cycle:
            bkgd_df=gen_spk_data(num_syn,T=pattern_activity_duration)
            if train_specs['neuron_model_is']=='sst':
                desired_num_spks=target_rate
            elif train_specs['neuron_model_is']=='mst':
                desired_num_spks=np.random.poisson(lam=target_rate)
            precomputed_data=precompute_data_for_iteration(neu_paras,bkgd_df,current_weights)
            outspk_times,vMax,tMax,jMax=get_outspk_times(bkgd_df,neu_paras,*precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=get_threshcrossing_Max_only)
            precomputed_data=[var[:jMax+1] for var in precomputed_data]
            
            num_outspks=len(outspk_times)
            error_sign=np.sign(desired_num_spks-num_outspks)
            if error_sign:            
                if train_specs['learning_rule_is']=='corr_thresh' or train_specs['learning_rule_is']=='corr_top_p':                    
                    if train_specs['neuron_model_is']=='sst':
                        #pattern_df.loc[:jMax,'elig']=get_corr_inspk_eligibilities(neu_paras,pattern_df.loc[:jMax,'spk_ti1es'].values,outspk_times,*precomputed_data)#,np.zeros((len(pattern_df)-(jMax+1)))))
                        bkgd_df['elig']=np.concatenate((get_corr_inspk_eligibilities(neu_paras,bkgd_df.loc[:jMax,'spk_times'].values,outspk_times,*precomputed_data),np.zeros((len(bkgd_df)-(jMax+1)))))
                    elif train_specs['neuron_model_is']=='mst':
                        bkgd_df['elig']=get_corr_inspk_eligibilities(neu_paras,bkgd_df.spk_times.values,outspk_times,*precomputed_data)
                    
                neuron_eligibilities=get_neuron_eligibilities(bkgd_df,neu_paras)#,normalize=False,num_outspks=num_outspks,current_weights=current_weights)

                current_weight_change=get_weight_change(current_weights,error_sign,warmup_learning_rate,neuron_eligibilities,train_specs['learning_rule_is'])
                
                current_weights+=current_weight_change
                assert np.isnan(current_weights).sum()==0,'a weight is nan!'
            
            store_num_spks.append(num_outspks)
            it+=1
        if np.mean(store_num_spks)>target_rate:
            print('warmup took '+str(cycle)+' cycles, giving '+str(np.mean(store_num_spks))+'Hz')
            break 
        cycle += 1
    return current_weights,seed
  
#@profile 
def train_model(neu_paras,train_specs,initial_weight_std,n_cycles,learning_rate,seed,divfac=np.Inf,\
                trials_per_cycle=100,pattern_activity_duration=500,do_warmup=True,
                feature_data=None,input_data=None,use_momentum=True):
    seed+=1
    np.random.seed(seed)
    num_syn=neu_paras['num_syn']
    current_weights=np.random.normal(scale=initial_weight_std,size=num_syn)

    #set-up
    if train_specs['neuron_model_is']=='mst':
        if train_specs['labels_are']=='agg':
            features,fea_count_means,fea_labels=feature_data
        if train_specs['learn_from']=='teacher':
            teacher_initial_weight_std=1./np.sqrt(num_syn)
    elif train_specs['neuron_model_is']=='sst':
        if train_specs['learn_from']=='teacher':
            teacher_initial_weight_std=1./np.sqrt(num_syn)
        elif train_specs['learn_from']=='labeled_data':
            input_patterns,target_labels=input_data
            trials_per_cycle=len(input_patterns)
    else:
        exit('train_specs ill-defined')
        
    if train_specs['learning_rule_is']=='corr_thresh' or train_specs['learning_rule_is']=='corr_top_p':
        get_threshcrossing_Max_only=False
        if train_specs['learning_rule_is']=='corr_thresh':
            current_weights+=1e-2 #ensures plasticity induction threshold crossed
    elif train_specs['learning_rule_is']=='Vmax_grad':
        get_threshcrossing_Max_only=True
    print(train_specs['neuron_model_is']+' learns from '+train_specs['labels_are']+' '+train_specs['learn_from']+' using '+train_specs['learning_rule_is'])        
    if train_specs['learn_from']=='teacher':     #initialize teacher
        seed+=1
        np.random.seed(seed)
        teacher_weights=np.random.normal(scale=teacher_initial_weight_std,size=num_syn)
        seed+=1
        np.random.seed(seed)
        teacher_weights,seed=warmup(teacher_weights,trials_per_cycle,n_cycles,neu_paras,seed,train_specs,get_threshcrossing_Max_only)
        print('warmed up teacher')
    if do_warmup: #warmup (only appreciable speed-up for easy tasks/fast convergence
        seed+=1
        np.random.seed(seed)
        current_weights,seed=warmup(current_weights,trials_per_cycle,n_cycles,neu_paras,seed,train_specs,get_threshcrossing_Max_only)  
        print('warmed up learner')

    initial_weights_norm=np.linalg.norm(current_weights)
    print('initial weights: min '+str(np.min(current_weights))+' max '+str(np.max(current_weights))+' norm '+str(initial_weights_norm))
    
    #initialize storage containers
    neu_elig_Store= []
    cur_weights_list = [deepcopy(current_weights)]
    desired_numspkslist= []
    numspkslist =[]
    
    #initialize changing loop variables
    previous_weight_change=np.zeros(current_weights.shape)
    #grad_cache=np.zeros(current_weights.shape)    #for RMS prop speed-up below
    cycle = 0

    seed+=1
    np.random.seed(seed)
    performance=0.
    presentation_order=range(trials_per_cycle)
    while cycle < n_cycles and performance<trials_per_cycle:     
        st=time.time()
        performance=0.
        error_sign_Store=[]
        print_flag=True
        desired_numspkslist= []
        numspkslist =[]
        np.random.shuffle(presentation_order)
        for it in range(trials_per_cycle):
            
            #drive model with input to get output
            if train_specs['neuron_model_is']=='mst':
                if train_specs['learn_from']=='labeled_data':
                    pattern_df,occur_df=get_input_pattern(features,fea_count_means,num_syn)
                    desired_num_spks=0 if occur_df.empty else fea_labels[occur_df.occur_idx.values].sum()  #label-weighted aggregate number of clues in the pattern    
                elif train_specs['learn_from']=='teacher':
                    pattern_df=gen_spk_data(num_syn,T=pattern_activity_duration)
                    teacher_precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,teacher_weights)
                    teacher_outspk_times,_,_,_=get_outspk_times(pattern_df,neu_paras,*teacher_precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=get_threshcrossing_Max_only)
                    desired_num_spks=len(teacher_outspk_times)   
                precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,current_weights)
                outspk_times,_,_,_=get_outspk_times(pattern_df,neu_paras,*precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=get_threshcrossing_Max_only)
            elif train_specs['neuron_model_is']=='sst':
                if train_specs['learn_from']=='labeled_data':
                    pattern_df=deepcopy(input_patterns[presentation_order[it]])
                    desired_num_spks=target_labels[presentation_order[it]]
                elif train_specs['learn_from']=='teacher':
                    pattern_df=gen_spk_data(num_syn,T=pattern_activity_duration)
                    teacher_precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,teacher_weights)
                    outspk_times,vMax,tMax,jMax=get_outspk_times(pattern_df,neu_paras,*teacher_precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=get_threshcrossing_Max_only)
                    desired_num_spks=len(outspk_times)    
                precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,current_weights)
                outspk_times,vMax,tMax,jMax=get_outspk_times(pattern_df,neu_paras,*precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=get_threshcrossing_Max_only)
                #pattern_df=pattern_df[:jMax+1] #reallocation too costly (factor 3 slow down), so just use indexing when sending into function
                precomputed_data=[var[:jMax+1] for var in precomputed_data]
            num_outspks=len(outspk_times)
            
            #store classifier output activity
            desired_numspkslist.append(desired_num_spks)
            numspkslist.append(num_outspks)
     
            #learn
            error_sign=np.sign(desired_num_spks-num_outspks)
            if error_sign:
                                
                #compute inspike eligibility
                pattern_df['elig']=0
                if train_specs['learning_rule_is']=='STS_grad':
                    print('STS-gradient not implemented yet!')                        
                elif train_specs['learning_rule_is']=='Vmax_grad':
                    pattern_df.loc[:jMax+1,'elig'] = neu_paras['v_norm']*(np.exp((pattern_df.loc[:jMax+1,'spk_times']-tMax)/neu_paras['tau_mem']) \
                                                                        - np.exp((pattern_df.loc[:jMax+1,'spk_times']-tMax)/neu_paras['tau_syn']))
                elif train_specs['learning_rule_is']=='corr_thresh' or train_specs['learning_rule_is']=='corr_top_p':                    
                    if train_specs['neuron_model_is']=='sst':
                        #pattern_df.loc[:jMax,'elig']=get_corr_inspk_eligibilities(neu_paras,pattern_df.loc[:jMax,'spk_ti1es'].values,outspk_times,*precomputed_data)#,np.zeros((len(pattern_df)-(jMax+1)))))
                        pattern_df['elig']=np.concatenate((get_corr_inspk_eligibilities(neu_paras,pattern_df.loc[:jMax,'spk_times'].values,outspk_times,*precomputed_data),np.zeros((len(pattern_df)-(jMax+1)))))
                    elif train_specs['neuron_model_is']=='mst':
                        pattern_df['elig']=get_corr_inspk_eligibilities(neu_paras,pattern_df.spk_times.values,outspk_times,*precomputed_data)
                    
                neuron_eligibilities=get_neuron_eligibilities(pattern_df,neu_paras)#,normalize=False,num_outspks=num_outspks,current_weights=current_weights) 

                current_weight_change=get_weight_change(current_weights,error_sign,learning_rate,neuron_eligibilities,train_specs['learning_rule_is'])
                
                if use_momentum:
                    current_weight_change,previous_weight_change=momentum_step(current_weight_change,previous_weight_change)
                
                current_weights+=current_weight_change
                assert np.isnan(current_weights).sum()==0,'a weight is nan!'

                #RMS prop (would have to refactor as RMSprop_step function        
                #rms_decay_rate=0.99
                #grad_cache = rms_decay_rate* grad_cache + (1 - rms_decay_rate)*weight_change_tmp*weight_change_tmp
                #current_weights+= learning_rate*weight_change_tmp/(np.sqrt(grad_cache) + np.finfo(np.float).eps)
                
                #divisive normalization to stabilize learning at long times for hard tasks could also refractor in div_renorm_step
                current_norm=np.linalg.norm(current_weights)
                if current_norm>divfac*initial_weights_norm: 
                    if print_flag:
                        print('divisive normalization applied to at least one trial in this cycle')
                        print_flag=False
                    current_weights*=divfac*initial_weights_norm/current_norm
            else:
                performance+=1
            error_sign_Store.append(error_sign)
        
        if train_specs['learning_rule_is']=='corr_thresh': 
            print('affected:'+str((neuron_eligibilities>1e-3).sum()))
        
        neu_elig_Store.append(deepcopy(neuron_eligibilities))
        cur_weights_list.append(deepcopy(current_weights))
        
        cycle += 1
        et=time.time()
        
        recent_discrep=np.array(numspkslist[-trials_per_cycle:])-np.array(desired_numspkslist[-trials_per_cycle:])

        #cycle, time, number of correct trials,number of undershoot trials, number of overshoot trials,weight (min,avg,max), weight norm, eligibility (min,avg,max)
        print 'c: {0:4d} {1:3.6f} {2:s} {3:4d} {4:4d} {5:3.6f} {6:3.6f} {7:3.6f} {8:3.6f} {9:3.6f} {10:3.6f} {11:3.6f}' \
                  .format(cycle, et-st, str(int(100*performance/trials_per_cycle))+'%', \
                          np.sum(np.array(error_sign_Store)==1),np.sum(np.array(error_sign_Store)==-1),\
                          np.min(current_weights), np.mean(current_weights),np.max(current_weights), current_norm,\
                          np.min(neuron_eligibilities),np.mean(neuron_eligibilities),np.max(neuron_eligibilities))  
        if train_specs['neuron_model_is']=='mst':
            performance=0
        np.save('neuron_eli.npy',neu_elig_Store)
    if train_specs['learn_from']=='teacher':
        return cur_weights_list, desired_numspkslist,numspkslist,seed,teacher_weights
    else:
        return cur_weights_list, desired_numspkslist,numspkslist,seed

################0 Analyze learning
def get_rates(current_weights,feature_data,neu_paras,bkgd_activity_duration=2000.,num_probe_trials=1000,bkgd_avg_syn_firingrate=5*1/1000.):
    num_syn=len(current_weights)
    
    feature_duration=feature_data['ensemble_paras']['feature_duration']
    fea_patterns_tmp=deepcopy(feature_data['data'])
    for feature_df in fea_patterns_tmp:
        feature_df.spk_times+=bkgd_activity_duration/2.
        
    seed_rates=0
    num_spks_present =np.zeros((num_probe_trials,len(fea_patterns_tmp)))
    num_spks_absent  =np.zeros((num_probe_trials,len(fea_patterns_tmp)))
    while seed_rates <num_probe_trials:
        bkgd_df=gen_spk_data(num_syn, bkgd_avg_syn_firingrate, bkgd_activity_duration)
        bkgd_df.loc[bkgd_df.spk_times>bkgd_activity_duration/2.,'spk_times']+=feature_duration
         
        outspk_times_absent,vMax,tMax,jMax=get_outspk_times(bkgd_df,neu_paras,*precompute_data_for_iteration(neu_paras,bkgd_df,current_weights),neuron_model_is='mst')
                                             
        num_spks_absent[seed_rates,:]=len(outspk_times_absent)
        
        for fit,feature in enumerate(fea_patterns_tmp):
            pattern_df=bkgd_df.append(feature,ignore_index=True).sort_values(by='spk_times').reset_index(drop=True)
            outspk_times_present,vMax,tMax,jMax=get_outspk_times(pattern_df,neu_paras,*precompute_data_for_iteration(neu_paras,pattern_df,current_weights),neuron_model_is='mst')
            num_spks_present[seed_rates,fit] =len(outspk_times_present)
        seed_rates+=1
            
    return num_spks_present,num_spks_absent
    
def get_learning_curve_data(path,seed,neu_paras,Tprobe=2000):
    cur_weights_list=list(np.load(path+'cur_weights_list.npy'))
    features=np.load(path+'features.npy').item()
    seed+=1
    np.random.seed(seed)
    partial_get_rates=partial(get_rates,feature_data=features,bkgd_activity_duration=Tprobe,neu_paras=neu_paras)
    stepsize=int(len(cur_weights_list)/10)
    #run
    st=time.time()
    pool = Pool(processes=8)
    num_spks_present_iters,num_spks_absent_iters=zip(*pool.map(partial_get_rates,cur_weights_list[::stepsize]))
    pool.close()
    pool.join()
    et=time.time()
    print('learnign curve took '+str(et-st))
    
    #compute mean responses
    mean_present_rates =np.asarray([np.mean(num_spks_present,axis=0) for num_spks_present in num_spks_present_iters])
    mean_absent_rates  =np.asarray([np.mean(num_spks_absent ,axis=0) for num_spks_absent  in num_spks_absent_iters ])
    np.save(path+'excess_rates_stepsize_'+str(stepsize), mean_present_rates-mean_absent_rates)
    np.save(path+  'bkgd_rates_stepsize_'+str(stepsize), mean_absent_rates)#/Tprobe*(1000/1)) #convert to Hertz
    
    #store raw data
    np.save(path+'present_data_stepsize_'+str(stepsize),  num_spks_present_iters)
    np.save(path+ 'absent_data_stepsize_'+str(stepsize),  num_spks_absent_iters )
    return seed

def get_gen_error(weights_and_idx,teacher_weights,train_specs,neu_paras,pattern_activity_duration=500.,num_probe_trials=10000,pattern_avg_syn_firingrate=5*1/1000.):
    
    current_weights=weights_and_idx[1]
    seed=weights_and_idx[0]
    
    np.random.seed(seed)
    
    num_syn=len(current_weights)
    num_spks_student  =np.zeros(num_probe_trials)
    num_spks_teacher  =np.zeros(num_probe_trials)
    it=0
    while it <num_probe_trials:
        pattern_df=gen_spk_data(num_syn, pattern_avg_syn_firingrate, pattern_activity_duration)

        #teacher output:
        teacher_precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,teacher_weights)
        teacher_outspk_times,_,_,_=get_outspk_times(pattern_df,neu_paras,*teacher_precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=True)
        num_spks_teacher[it]=len(teacher_outspk_times)
        
        #output:
        precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,current_weights)
        outspk_times,_,_,_=get_outspk_times(pattern_df,neu_paras,*precomputed_data,neuron_model_is=train_specs['neuron_model_is'],get_threshcrossing_Max_only=True)
        num_spks_student[it]=len(outspk_times)
        
        it+=1
                
    return num_spks_teacher,num_spks_student 
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    