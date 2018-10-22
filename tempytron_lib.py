#pkgs/utilities
import numpy as np
import pandas as pd
from functools import partial
from copy import deepcopy
import time
#############################0 pattern generation
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

############################0 model parameters  
def gen_neuron_paras(tau_syn=5.,tau_mem=20.,v_thresh=1.): #parameter dictionary passed to functions
    eta=tau_mem/tau_syn
    v_norm=eta**(eta/(eta-1))/(eta-1)
    neu_paras={
        'tau_syn':tau_syn,
        'tau_mem':tau_mem,
        'v_norm':v_norm,
        'v_thresh':v_thresh
        }
    return neu_paras

########################0 getting output spike times  
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
    told = 0; tnew = tlower#(tlower+tupper)/2 
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
  
def get_outspk_times(pattern_df,neu_paras,logDvec,logDsvec,sgnDvec,sgnDsvec,output_voltage=False):
    outspk_times=list()
    tau_mem=neu_paras['tau_mem']
    tau_syn=neu_paras['tau_syn']
    v_norm=neu_paras['v_norm']
    v_thresh=neu_paras['v_thresh']
    eta=tau_mem/tau_syn
    logeta=np.log(eta)
    taufrac=tau_mem/(1-eta)
    inspk_times=np.append(pattern_df.spk_times.values,np.Inf)
    inspk_times_tau_mem=inspk_times/tau_mem
    inspk_times_tau_syn=inspk_times/tau_syn
    logDvec_mod=deepcopy(logDvec)#modified versions contain effects of any output spikes. Used to get voltage values at input spike times
    sgnDvec_mod=deepcopy(sgnDvec)
    for j,inspk_time in enumerate(inspk_times[:-1]): #don't include appended inf spiketime
        if sgnDvec_mod[j]>0 and sgnDsvec[j]>0: #maximum (in absence of future spikes)?  1
            tMaxcheck = (logDvec_mod[j]-logDsvec[j]-logeta)*taufrac
            if (tMaxcheck>inspk_time and tMaxcheck<inspk_times[j+1]): #maximum in this inter spike interval?
                vMaxcheck = np.exp(logDsvec[j]/(1-eta)+logDvec_mod[j]/(1-1/eta)) #height of interior maximum
                t_upper=tMaxcheck
            else: #when maximum is (before or) after next spike time
                if tMaxcheck>inspk_times[j+1]: #if after, then calc value at next input spike
                    vMaxcheck=v_norm*(np.exp(logDvec_mod[j]-inspk_times_tau_mem[j+1])-np.exp(logDsvec[j]-inspk_times_tau_syn[j+1])) #voltage at next input spike time  
		    t_upper=inspk_times[j+1]
		else:
		    vMaxcheck=0 #maximum is before current input spike so ignore.
            
            if vMaxcheck>v_thresh: #threshold is crossed: 
		Dvec_mod_tmp=sgnDvec_mod[j]*np.exp(logDvec_mod[j])
		sgnDvec_mod_tmp=np.sign(Dvec_mod_tmp)
		logDvec_mod_tmp=np.log(np.fabs(Dvec_mod_tmp))
		vdv_rootfunc=partial(get_vdv_rootfunc,logD=logDvec_mod_tmp,logDs=logDsvec[j],sgnD=sgnDvec_mod_tmp,sgnDs=sgnDsvec[j],neuron_paras=neu_paras)
		spike_term=0
		tlower=deepcopy(inspk_time)
		while vMaxcheck>v_thresh: # or interior_maximum_cond  #run root finding iteratively until voltage at next spike is subthreshold and no interior maximum
		    outspk_time=root_finding(tlower,t_upper,vdv_rootfunc)
		    outspk_times.append(outspk_time)
		    spike_term+=-v_thresh/v_norm*np.exp(outspk_time/tau_mem)
		    Dvec_mod_tmp=sgnDvec_mod[j]*np.exp(logDvec_mod[j])+spike_term
		    sgnDvec_mod_tmp=np.sign(Dvec_mod_tmp)
		    logDvec_mod_tmp=np.log(np.fabs(Dvec_mod_tmp))
		    vdv_rootfunc=partial(get_vdv_rootfunc,logD=logDvec_mod_tmp,logDs=logDsvec[j],sgnD=sgnDvec_mod_tmp,sgnDs=sgnDsvec[j],neuron_paras=neu_paras)
		    vMaxcheck=vdv_rootfunc(t_upper)[0]+v_thresh
		    tlower=outspk_time
		    
		#update total interval contribution
		Dvec_mod_tmp=sgnDvec_mod[j+1:]*np.exp(logDvec_mod[j+1:])+spike_term
		sgnDvec_mod[j+1:]=np.sign(Dvec_mod_tmp)
		logDvec_mod[j+1:]=np.log(np.fabs(Dvec_mod_tmp))
		
    if output_voltage:  #used to check with numerical solution 
	voltage_at_spikes = v_norm*(sgnDvec_mod*np.exp(logDvec_mod-inspk_times_tau_mem[:-1]) \
	                             - sgnDsvec*np.exp(logDsvec   -inspk_times_tau_syn[:-1]))
        return np.asarray(outspk_times), voltage_at_spikes
    else:
        return np.asarray(outspk_times)

def precompute_data_for_iteration(neuron_paras,pattern_df,weights):
    w_seq=weights[pattern_df.spk_idx]
    logw_seq=np.log(np.fabs(w_seq))
    #Dvec= np.cumsum(w_seq*np.exp(pattern_df.spk_times.values/neuron_paras['tau_mem']))
    #Dsvec=np.cumsum(w_seq*np.exp(pattern_df.spk_times.values/neuron_paras['tau_syn']))
    Dvec= np.cumsum(np.sign(w_seq)*np.exp(logw_seq+pattern_df.spk_times.values/neuron_paras['tau_mem']))
    Dsvec=np.cumsum(np.sign(w_seq)*np.exp(logw_seq+pattern_df.spk_times.values/neuron_paras['tau_syn']))
    logDvec=np.log(np.fabs(Dvec))
    logDsvec=np.log(np.fabs(Dsvec))
    sgnDvec=np.sign(Dvec)
    sgnDsvec=np.sign(Dsvec)
    return logDvec,logDsvec,sgnDvec,sgnDsvec
  
############0 learn
def get_inspk_eligibilities(neuron_paras,inspk_times,outspk_times,logDvec,logDsvec,sgnDvec,sgnDsvec):
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
        
    out_spk_after=(inspk_times[:-1,np.newaxis]<outspk_times[np.newaxis,:])
    exp_out_mem=np.exp((inspk_times[:-1,np.newaxis]-outspk_times[np.newaxis,:])/tau_mem)
    exp_out_syn=np.exp((inspk_times[:-1,np.newaxis]-outspk_times[np.newaxis,:])/tau_syn)
    
    reset_corr=-neuron_paras['v_norm']*neuron_paras['v_thresh']*tau_mem/2* \
          np.sum( ((tau_mem-tau_syn)/(tau_mem+tau_syn))*((~out_spk_after)/exp_out_mem) \
	         + (exp_out_mem-(2*tau_syn/(tau_mem+tau_syn))*exp_out_syn)*out_spk_after \
	        ,axis=1)
    return reset_corr + kernel_corr

def corr_weight_update(learning_rate,learning_sgn,top_elig_fraction,eligibilities,old_weights):
    old_weights[np.argsort(eligibilities)[:int(top_elig_fraction*len(old_weights))]]+=learning_sgn*learning_rate   
    return old_weights
 
def correlation_training(neu_paras,feature_data,current_weights, initial_weights,fea_count_means, fea_labels, n_cycles, learning_rate_elig,top_elig_fraction,momentum_factor,divfac,seed):
    num_syn=len(current_weights)
    trials_per_cycle=100
    
    if (current_weights==initial_weights).all(): #warmup (only appreciable speed-up for easy tasks/fast convergence
        current_weights,seed=warmup(current_weights,top_elig_fraction,trials_per_cycle,n_cycles,num_syn,neu_paras,seed)
        initial_weights=deepcopy(current_weights)
        print('initial weights: '+str(np.min(current_weights))+' '+str(np.min(current_weights)))
    print('starting weights: '+str(np.min(current_weights))+' '+str(np.min(current_weights)))
    neu_elig_Store= []
    cur_weights_list = []
    old_weight_change=np.zeros(current_weights.shape)
    #grad_cache=np.zeros(current_weights.shape)    #for RMS prop speed-up below
    initial_weights_norm=np.linalg.norm(initial_weights)
    cycle = 0
    seed+=1
    np.random.seed(seed)
    it=0
    while cycle < n_cycles:     
        trial = cycle*trials_per_cycle
        st=time.time()
        while it < trial+trials_per_cycle:
	    pattern_df,occur_df=get_input_pattern(feature_data,fea_count_means,num_syn)
            
            desired_num_spks=0 if occur_df.empty else fea_labels[occur_df.occur_idx.values].sum()  #label-weighted aggregate number of clues in the pattern
	    
	    precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,current_weights)
	    
	    #output:
            outspk_times=get_outspk_times(pattern_df,neu_paras,*precomputed_data)
            num_spks=len(outspk_times)
            
            error_sign=np.sign(desired_num_spks-num_spks)
            if error_sign:
                #compute eligibiliity
                pattern_df['elig']=get_inspk_eligibilities(neu_paras,pattern_df.spk_times.values,outspk_times,*precomputed_data)
                eli_tmp=pattern_df.groupby('spk_idx').elig.sum()
                neuron_eligibilities=np.zeros(num_syn)
                neuron_eligibilities[eli_tmp.index.values]=eli_tmp.values
		neuron_eligibilities[np.isnan(neuron_eligibilities)]=0.
		
                #weight update
                old_weights=deepcopy(current_weights)
                current_weights=corr_weight_update(learning_rate_elig,error_sign,top_elig_fraction,neuron_eligibilities,current_weights)
                #x=np.exp(2*(neuron_eligibilities-np.mean(neuron_eligibilities))/np.fabs(np.mean(neuron_eligibilities)))-5
                #current_weights+=learning_rate_elig*error_sign*np.exp(x)/np.sum(np.exp(x))#*10
                if np.isnan(current_weights).sum()>0:
		    #print(len(eli_tmp))
		    #print(x)
		    #print(old_weights)
		    print('a weight is nan! Set to 0')
		    current_weights[np.isnan(current_weights)]=0.
		    #exit()
                
                weight_change_tmp=current_weights-old_weights

                ##momentum speed-up:
                current_weights+=momentum_factor*old_weight_change*(weight_change_tmp!=0)  #only add momentum to finite weight changes
                old_weight_change[weight_change_tmp!=0]=(current_weights-old_weights)[weight_change_tmp!=0]

                #RMS prop
                #rms_decay_rate=0.99
                #grad_cache = rms_decay_rate* grad_cache + (1 - rms_decay_rate)*weight_change_tmp*weight_change_tmp
                #current_weights+= learning_rate_elig*weight_change_tmp/(np.sqrt(grad_cache) + np.finfo(np.float).eps)
                
                #divisive normalization to stabilize learning at long times for hard tasks
                current_norm=np.linalg.norm(current_weights)
                if current_norm>divfac*initial_weights_norm: 
                    print('divisive normalization applied')
                    current_weights*=divfac*initial_weights_norm/current_norm

            it += 1
        #neuron_eligibilities=10*np.exp(x)/np.sum(np.exp(x))
	#print('elig: '+ str(np.min(neuron_eligibilities))+' '+str(np.mean(neuron_eligibilities))+' '+str(np.max(neuron_eligibilities)))
	#neu_elig_Store.append(deepcopy(neuron_eligibilities))
        cur_weights_list.append(deepcopy(current_weights))
        cycle += 1
        et=time.time()
        print('c: '+str(cycle)+' '+str(et-st))
        np.save('neuron_eli.npy',neu_elig_Store)
    return cur_weights_list,seed

def warmup(current_weights,top_elig_fraction,trials_per_cycle,n_cycles,num_syn,neu_paras,seed):
    seed+=1
    np.random.seed(seed)
    cycle = 0
    it=0
    while cycle < n_cycles: 
	trial = cycle*trials_per_cycle
	store_num_spks=[]
	while it < trial+trials_per_cycle:
	    bkgd_df=gen_spk_data(num_syn)
	    desired_num_spks=np.random.poisson(lam=5)
	    
	    precomputed_data=precompute_data_for_iteration(neu_paras,bkgd_df,current_weights)

	    outspk_times=get_outspk_times(bkgd_df,neu_paras,*precomputed_data)
	    num_spks=len(outspk_times)

	    error_sign=np.sign(desired_num_spks-num_spks)
	    
	    if error_sign:
		#compute eligibiliity
		bkgd_df['elig']=get_inspk_eligibilities(neu_paras,bkgd_df.spk_times.values,outspk_times,*precomputed_data)
		elitmp=bkgd_df.groupby('spk_idx').elig.sum()
		neuron_eligibilities=np.zeros((num_syn,))
		neuron_eligibilities[elitmp.index.values]=elitmp.values
		#weight update
		current_weights=corr_weight_update(1e-3,error_sign,top_elig_fraction,neuron_eligibilities,current_weights)
	    store_num_spks.append(num_spks)
	    it+=1
	    
	if np.mean(store_num_spks)>5.:
	    print('warmup took '+str(cycle)+' cycles, giving '+str(np.mean(store_num_spks))+'Hz')
	    break 
	cycle += 1
    return current_weights,seed
	    
	    
################0 Analyze learning
def get_rates(current_weights,feature_data,neuron_paras,bkgd_activity_duration=2000.,num_probe_trials=1000,bkgd_avg_syn_firingrate=5*1/1000.):
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

	outspk_times_absent     =get_outspk_times(bkgd_df,   neuron_paras,*precompute_data_for_iteration(neuron_paras,   bkgd_df,current_weights))
	num_spks_absent[seed_rates,:]=len(outspk_times_absent)
	
	for fit,feature in enumerate(fea_patterns_tmp):
	    pattern_df=bkgd_df.append(feature,ignore_index=True).sort_values(by='spk_times').reset_index(drop=True)
	    outspk_times_present=get_outspk_times(pattern_df,neuron_paras,*precompute_data_for_iteration(neuron_paras,pattern_df,current_weights))
	    num_spks_present[seed_rates,fit] =len(outspk_times_present)
	seed_rates+=1
	    
    return num_spks_present,num_spks_absent
    
    
####################0 Student Teacher learning

def student_teacher_training(neu_paras,current_weights, initial_weights, n_cycles, learning_rate_elig,top_elig_fraction,momentum_factor,divfac,seed):
    num_syn=len(current_weights)
    trials_per_cycle=100
    
    #initialize teacher
    seed+=1
    np.random.seed(seed) 
    teacher_weights=np.random.normal(scale=1./np.sqrt(num_syn),size=num_syn)
    #warm-up teacher
    teacher_weights=warmup(teacher_weights,top_elig_fraction,trials_per_cycle,n_cycles,num_syn,neu_paras,seed)
    
    #warm-up student
    seed+=1
    initial_weights=warmup(current_weights,top_elig_fraction,trials_per_cycle,n_cycles,num_syn,neu_paras,seed)

    print('initial weights: '+str(np.min(current_weights))+' '+str(np.min(current_weights)))
    neu_elig_Store= []
    cur_weights_list = []
    old_weight_change=np.zeros(current_weights.shape)
    desired_numspkslist= []
    numspkslist =[]
    
    #grad_cache=np.zeros(current_weights.shape)    #for RMS prop speed-up below
    initial_weights_norm=np.linalg.norm(initial_weights)
    cycle = 0
    seed+=1
    np.random.seed(seed)
    it=0
    while cycle < n_cycles:     
        trial = cycle*trials_per_cycle
        st=time.time()
        while it < trial+trials_per_cycle:

	    #pattern:
            pattern_df=gen_spk_data(num_syn,T=500)
            
	    #teacher output:
	    teacher_precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,teacher_weights)
	    teacher_outspk_times=get_outspk_times(pattern_df,neu_paras,*teacher_precomputed_data)
	    desired_num_spks=len(teacher_outspk_times)
	    
	    #output:
	    precomputed_data=precompute_data_for_iteration(neu_paras,pattern_df,current_weights)
            outspk_times=get_outspk_times(pattern_df,neu_paras,*precomputed_data)
            num_spks=len(outspk_times)
            
            desired_numspkslist.append(desired_num_spks)
            numspkslist.append(num_spks)
            
            error_sign=np.sign(desired_num_spks-num_spks)
            if error_sign:
                #compute eligibiliity
                pattern_df['elig']=get_inspk_eligibilities(neu_paras,pattern_df.spk_times.values,outspk_times,*precomputed_data)
                eli_tmp=pattern_df.groupby('spk_idx').elig.sum()
                neuron_eligibilities=np.zeros(num_syn)
                neuron_eligibilities[eli_tmp.index.values]=eli_tmp.values
		neuron_eligibilities[np.isnan(neuron_eligibilities)]=0.
		
                #weight update
                old_weights=deepcopy(current_weights)
                current_weights=corr_weight_update(learning_rate_elig,error_sign,top_elig_fraction,neuron_eligibilities,current_weights)
                #x=np.exp(2*(neuron_eligibilities-np.mean(neuron_eligibilities))/np.fabs(np.mean(neuron_eligibilities)))-5
                #current_weights+=learning_rate_elig*error_sign*np.exp(x)/np.sum(np.exp(x))#*10
                if np.isnan(current_weights).sum()>0:
		    #print(len(eli_tmp))
		    #print(x)
		    #print(old_weights)
		    print('a weight is nan! Set to 0')
		    current_weights[np.isnan(current_weights)]=0.
		    #exit()
                
                weight_change_tmp=current_weights-old_weights

                ##momentum speed-up:
                current_weights+=momentum_factor*old_weight_change*(weight_change_tmp!=0)  #only add momentum to finite weight changes
                old_weight_change[weight_change_tmp!=0]=(current_weights-old_weights)[weight_change_tmp!=0]

                #RMS prop
                #rms_decay_rate=0.99
                #grad_cache = rms_decay_rate* grad_cache + (1 - rms_decay_rate)*weight_change_tmp*weight_change_tmp
                #current_weights+= learning_rate_elig*weight_change_tmp/(np.sqrt(grad_cache) + np.finfo(np.float).eps)
                
                #divisive normalization to stabilize learning at long times
                current_norm=np.linalg.norm(current_weights)
                if current_norm>divfac*initial_weights_norm:  #seems to call too early for some not yet understood reason... 
                    #print('divisive normalization applied')
                    current_weights*=divfac*initial_weights_norm/current_norm

            it += 1
        #neuron_eligibilities=10*np.exp(x)/np.sum(np.exp(x))
	#print('elig: '+ str(np.min(neuron_eligibilities))+' '+str(np.mean(neuron_eligibilities))+' '+str(np.max(neuron_eligibilities)))
	#neu_elig_Store.append(deepcopy(neuron_eligibilities))
        cur_weights_list.append(deepcopy(current_weights))
        cycle += 1
        et=time.time()
        recent_discrep=np.array(numspkslist[-trials_per_cycle:])-np.array(desired_numspkslist[-trials_per_cycle:])
        print('c: '+str(cycle)+' '+str(et-st)+' '+str(np.min(current_weights))+' '+str(np.max(current_weights))+' '+str(np.std(recent_discrep)))
        np.save('neuron_eli.npy',neu_elig_Store)
    return cur_weights_list, teacher_weights, desired_numspkslist,numspkslist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    