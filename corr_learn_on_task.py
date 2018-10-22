#pkgs/utilities
import numpy as np
import pandas as pd
from functools import partial
from copy import deepcopy
import os
import time
import sys
from multiprocessing import Pool

import tempytron_lib
reload(tempytron_lib)
from tempytron_lib import gen_neuron_paras, correlation_training,gen_spk_data,get_rates

#profiling: with kernprof.py in directory, place @profile above any functions to be profiled , then run:
# kernprof -l tempytron_main.py
# python -m line_profiler tempytron_main.py.lprof > outnew2.txt #prints .lprof file output to text

if __name__ == "__main__":    
        
    #feature labels (0:distractor,>0:clue, non-distinct labels group features into a single clue)
    #fea_labels=np.array([0,0,0,0,1])           #easy task
    #fea_labels=np.array([1,2,3,4,5])           #hard task
    #fea_labels=np.array([1,2,3,4,5,0,0,0,0,0]) #hard task
    
    fea_labels=np.array([int(var) for var in sys.argv[1:]])

    #paras that change
    count_mean=2.
    learning_rate_elig=1e-5
    divfac=2#np.Inf

    #fixed paras
    num_fea=len(fea_labels)
    fea_count_means=count_mean*np.ones(num_fea)  #homogeneous across features
    num_syn=500
    neu_paras=gen_neuron_paras()
    n_cycles=1000
    top_elig_fraction=0.1 #most eligible fraction of weights to update
    momentum_factor=0.99
    
    
    #batchname='v1_momentum'
    #batchname='v2_momentum_cf_'+str(int(count_mean))+'_lr_4_norm'
    #batchname='v2_momentum_fix_cf_'+str(int(count_mean))+'_lr_4_norm'
    #batchname='v2_momentum_expeli_cf_'+str(int(count_mean))+'_lr_4_norm'
    #batchname='v2_momentum_expeli_p10_test10_cf_'+str(int(count_mean))+'_lr_4_norm'
    #batchname='v2_momentum_expeli_try2_cf_'+str(int(count_mean))+'_lr_4_norm'
    #batchname='v2_momentum_expeli_try2_cf_'+str(int(count_mean))+'_lr_5_norm'
    #batchname='v2_momentum_expeli_no10fac_testwarmup_cf_'+str(int(count_mean))+'_lr_4_norm'
    batchname='v2_momentum_condfix_cf_'+str(int(count_mean))+'_lr_'+str(int(-np.log10(learning_rate_elig)))+'_df_'+str(divfac)

    #batchname='v2_RMSprop

    #load any existing data
    batchname+='_labels_'+('_'.join(list(fea_labels.astype(str))))+'_'
    
    outpath='data/'
    if os.path.exists(outpath+batchname+'cur_weights_list.npy'):
        cur_weights_list=list(np.load(outpath+batchname+'cur_weights_list.npy'))
        initial_weights=cur_weights_list[0]
        current_weights=cur_weights_list[-1]
	seed=100#len(cur_weights_list)-1
	print(str(seed+1)+' existing iterations')
        feature_data=np.load(outpath+batchname+'feature_data.npy').item()
    else:
	ens_para_dict={'feature_duration':50.,'avg_syn_firingrate':5*(1/1000.)} #convert to Hz to /ms
	seed = 0
	np.random.seed(seed) #reserve integer-valued seeds for pattern generation
	fea_patterns=[gen_spk_data(num_syn, ens_para_dict['avg_syn_firingrate'], ens_para_dict['feature_duration']) for fea in range(num_fea)] #as a list of dataframes
        feature_data={'data':fea_patterns,'ensemble_paras':ens_para_dict}
        np.save(outpath+batchname+'feature_data',feature_data)
        weight_std=1./np.sqrt(num_syn)
        seed+=1
        np.random.seed(seed)
        current_weights=np.random.normal(scale=weight_std,size=num_syn)
        initial_weights=deepcopy(current_weights)
    
    learn=True
    if learn:
	st=time.time()
	weights_list,seed=correlation_training(neu_paras,feature_data,current_weights, initial_weights, fea_count_means, fea_labels,n_cycles, learning_rate_elig, top_elig_fraction,momentum_factor,divfac,seed)
	et=time.time()
	if os.path.exists(outpath+batchname+'cur_weights_list.npy'):
	    cur_weights_list=cur_weights_list+weights_list
	    np.save(outpath+batchname+'cur_weights_list',cur_weights_list)
	else:
	    np.save(outpath+batchname+'cur_weights_list',weights_list)
	print('learning took:'+str(et-st))
    
    track_learning=True
    if track_learning:
        cur_weights_list=list(np.load(outpath+batchname+'cur_weights_list.npy'))
        feature_data=np.load(outpath+batchname+'feature_data.npy').item()
        Tprobe=2000.
        seed+=1
        np.random.seed(seed)
        partial_get_rates=partial(get_rates,feature_data=feature_data,bkgd_activity_duration=Tprobe,neuron_paras=neu_paras)
	stepsize=int(len(cur_weights_list)/20)	
	
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
        np.save(outpath+batchname+'excess_rates_stepsize_'+str(stepsize), mean_present_rates-mean_absent_rates)
	np.save(outpath+batchname+  'bkgd_rates_stepsize_'+str(stepsize), mean_absent_rates)#/Tprobe*(1000/1)) #convert to Hertz
	
	#store raw data
	np.save(outpath+batchname+'present_data_stepsize_'+str(stepsize),  num_spks_present_iters)
	np.save(outpath+batchname+ 'absent_data_stepsize_'+str(stepsize),  num_spks_absent_iters )

	

    