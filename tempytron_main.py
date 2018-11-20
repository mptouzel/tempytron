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
from tempytron_lib import gen_neuron_paras, train_model,gen_spk_data,get_learning_curve_data,make_feature_data,get_gen_error

#profiling: 
#with kernprof.py in directory, place @profile above any functions to be profiled , then run:
# kernprof -l tempytron_main.py
#then generate readable output by running:
# python -m line_profiler tempytron_main.py.lprof > profiling_stats.txt 

if __name__ == "__main__":    

    #input options:

    #train_specs={'learn_from': sys.var[1],       #labeled_data,teacher
                 #'labels_are': sys.var[2],       #binary, aggregate
                 #'neuron_model_is': sys.var[3],  #sst,mst
                 #'learning_rule_is': sys.var[4]} #corr_thresh,corr_top_p,STS_grad,Vmax_grad
    
    #2006 paper:
    #train_specs={'neuron_model_is':'sst','labels_are':'binary','learn_from':'labeled_data','learning_rule_is':'corr_thresh'}
    #train_specs={'neuron_model_is':'sst','labels_are':'binary','learn_from':'labeled_data','learning_rule_is':'Vmax_grad'}
    train_specs={'neuron_model_is':'sst','labels_are':'binary','learn_from':'labeled_data','learning_rule_is':'corr_top_p'} #apply 2016 corr learning method to 2006 sst setting

    #2016 paper
    train_specs={'neuron_model_is':'mst','labels_are':'agg','learn_from':'labeled_data','learning_rule_is':'corr_top_p'}

        
    run_name='v1_momentum'
    ##############################################000> Run
    
    for neuron_model,learning_rule in zip(('sst','mst'),('Vmax_grad','STS_grad')):
        assert (train_specs['neuron_model_is']==neuron_model if train_specs['learning_rule_is']==learning_rule else True), learning_rule+'only for'+neuron_model+'!'

    outpath='data/'
    seed=0
    
    neu_paras=gen_neuron_paras()
    
    if train_specs['learn_from']=='labeled_data':

        if train_specs['labels_are']=='binary':
            neu_paras['tau_mem']=15.
            neu_paras['tau_syn']=neu_paras['tau_mem']/4            
            pattern_activity_duration=500
            n_patterns=2*neu_paras['num_syn']
            if train_specs['learning_rule_is']=='Vmax_grad':
                learning_rate=1e-4/neu_paras[   'v_norm']
            elif train_specs['learning_rule_is']=='corr_thresh' or train_specs['learning_rule_is']=='corr_top_p':
                learning_rate=8e-5
            n_cycles=500
            initial_weight_std=1e-3
            trials_per_cycle=n_patterns
            
            batchname='_'.join(train_specs.values())+'_'+runname+'_lr_'+str(int(-np.log10(learning_rate)))+'_T_'+str(pattern_activity_duration)+'_nc_'+str(n_cycles)+'_'            

            n_trials=1
            for trial in range(n_trials):
                st=time.time()
                seed=trial
                np.random.seed(seed)
                ens_para_dict={'feature_duration':pattern_activity_duration,'avg_syn_firingrate':5*(1/1000.)}
                input_patterns=[gen_spk_data(neu_paras['num_syn'], ens_para_dict['avg_syn_firingrate'], ens_para_dict['feature_duration']) for fea in range(n_patterns)]
                target_labels=np.random.rand(n_patterns)>0.5
                input_data=(input_patterns,target_labels)

                st=time.time()
                cur_weights_list, desired_numspkslist,numspkslist,seed=train_model( \
                                        neu_paras,train_specs,initial_weight_std,n_cycles,learning_rate,seed,input_data=input_data)
                et=time.time()
                np.save(outpath+batchname+'cur_weights_list_tr_'+str(trial),cur_weights_list)

                et=time.time()
                print('output: '+batchname)
                print('learning trial '+str(trial)+' took '+str(et-st))
                    
        elif train_specs['labels_are']=='agg':  
            runname='v1_testmomentum'
            #build data using feature labels (0:distractor,>0:clue, non-distinct labels group features into a single clue)
            fea_labels=np.array([1,2,3,4,5,0,0,0,0,0]) #hard task
            n_cycles=1000
            #fea_labels=np.array([1,0,0,0,0,0,0,0,0,0]) #easier task
            #n_cycles=200

            
            count_mean=2.
            num_fea=len(fea_labels)
            fea_count_means=count_mean*np.ones(num_fea)  #homogeneous across features
            feature_data,seed=make_feature_data(seed,neu_paras,num_fea,fea_count_means,fea_labels)
            learning_rate=1e-4
            divfac=5
            n_cycles=200
            initial_weight_std=2e-2#1./np.sqrt(neu_paras['num_syn'])   

            batchname='_'.join(train_specs.values())+'_'+runname+'_lr_'+str(int(-np.log10(learning_rate)))+'_cf_'+str(int(count_mean))+'_df_'+str(divfac)
            batchname+='_labels_'+('_'.join(list(fea_labels.astype(str))))+'_' 
            np.save(outpath+batchname+'features',feature_data[0])
            
            learn=True
            if learn:
                st=time.time()
                weights_list, desired_numspkslist,numspkslist,seed=train_model( \
                                        neu_paras,train_specs,initial_weight_std,n_cycles,learning_rate,seed,divfac=divfac,feature_data=feature_data)
                et=time.time()
                np.save(outpath+batchname+'cur_weights_list',weights_list)
                print('learning took:'+str(et-st))
                
            track_learning=True
            if track_learning:
                seed=get_learning_curve_data(outpath+batchname,seed,neu_paras)
    
    elif train_specs['learn_from']=='teacher':
            
        if train_specs['labels_are']=='agg':
            learning_rate=1e-4
            divfac=5
            n_cycles=500
            initial_weight_std=1./np.sqrt(neu_paras['num_syn'])            
        elif train_specs['labels_are']=='binary':
            neu_paras['tau_mem']=15.
            neu_paras['tau_syn']=neu_paras['tau_mem']/4
            pattern_activity_duration=500
            learning_rate=1e-4/neu_paras['v_norm']
            divfac=np.Inf
            n_cycles=2000
            initial_weight_std=1e-3
            
        batchname='_'.join(train_specs.values())+'_'+runname+'_lr_'+str(int(-np.log10(learning_rate)))+'_df_'+str(divfac)+'_T_'+str(pattern_activity_duration) \
                                        +'_nc_'+str(n_cycles)+'_'
        learn=True
        if learn:
            n_trials=1
            for trial in range(n_trials):
                st=time.time()
                seed=trial
                np.random.seed(seed) 
                st=time.time()
                cur_weights_list,desired_numspkslist,numspkslist,seed,teacher_weights=train_model(neu_paras,train_specs,initial_weight_std,n_cycles, \
                                                                                              learning_rate,seed,divfac=divfac)
                et=time.time()
                np.save(outpath+batchname+'cur_weights_list_tr_'+str(trial),cur_weights_list)
                np.save(outpath+batchname+'teacher_weights_tr_'+str(trial),teacher_weights)    
                np.save(outpath+batchname+'desired_spkslist_tr_'+str(trial),desired_numspkslist)
                np.save(outpath+batchname+'numspkslist_tr_'+str(trial),numspkslist)
                et=time.time()
                print('learning trial '+str(trial)+' took '+str(et-st))
                
        test=True
        if test:
            cur_weights_list=list(np.load(outpath+batchname+'cur_weights_list_tr_0.npy'))
            teacher_weights=np.load(outpath+batchname+'teacher_weights_tr_0.npy')    
            num_probe_trials=10000
            seed+=1
            np.random.seed(seed)
            partial_get_gen_error=partial(get_gen_error,teacher_weights=teacher_weights,neuron_paras=neu_paras,\
                                          pattern_activity_duration=pattern_activity_duration, num_probe_trials=num_probe_trials)
            stepsize=int(len(cur_weights_list)/20)
            #run
            st=time.time()
            pool = Pool(processes=8)
            num_spks_teacher_iters,num_spks_student_iters=zip(*pool.map(partial_get_gen_error,cur_weights_list[::stepsize]))
            pool.close()
            pool.join()
            et=time.time()
            print('learnign curve took '+str(et-st))
            #store raw data
            np.save(outpath+batchname+'student_data_tr_1_'+str(num_probe_trials)+'stepsize_'+str(stepsize),num_spks_student_iters)
            np.save(outpath+batchname+'teacher_data_tr_1_'+str(num_probe_trials)+'stepsize_'+str(stepsize),num_spks_teacher_iters)
            
            #compute mean responses
            if teacher_output_are_binary_values:
                gen_error =np.asarray([np.mean(num_spks_student*num_spks_teacher>0) \
                                      for (num_spks_teacher,num_spks_student) in zip(num_spks_teacher_iters,num_spks_student_iters)])
            else:
                gen_error =np.asarray([np.mean(np.power(num_spks_student-num_spks_teacher,2)) \
                                      for (num_spks_teacher,num_spks_student) in zip(num_spks_teacher_iters,num_spks_student_iters)])
            np.save(outpath+batchname+'gen_error_tr_1_np_'+str(num_probe_trials)+'_stepsize_'+str(stepsize), gen_error)

