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
from tempytron_lib import gen_neuron_paras, student_teacher_training,gen_spk_data,get_rates

#profiling: with kernprof.py in directory, place @profile above any functions to be profiled , then run:
# kernprof -l tempytron_main.py
# python -m line_profiler tempytron_main.py.lprof > outnew2.txt #prints .lprof file output to text

if __name__ == "__main__":    
        
    num_syn=500
    neu_paras=gen_neuron_paras()
    n_cycles=1000
    learning_rate_elig=1e-4
    divfac=5
    top_elig_fraction=0.1 #most eligible fraction of weights to update
    momentum_factor=0.99
    
    batchname='v1_st_momentum'
    batchname='v2_st_momentum_lr_'+str(int(-np.log10(learning_rate_elig)))+'_df_'+str(divfac)

    #batchname='v2_RMSprop
    
    outpath='data/'
    
    weight_std=1./np.sqrt(num_syn)
    seed = 0
    np.random.seed(seed) 
    current_weights=np.random.normal(scale=weight_std,size=num_syn)
    initial_weights=deepcopy(current_weights) 
    
    st=time.time()
    weights_list,teacher_weights,desired_numspkslist,numspkslist=student_teacher_training(neu_paras,current_weights, initial_weights,n_cycles, learning_rate_elig, top_elig_fraction,momentum_factor,divfac,seed)
    et=time.time()
    cur_weights_list=weights_list
    np.save(outpath+batchname+'cur_weights_list',cur_weights_list)
    np.save(outpath+batchname+'teacher_weights',teacher_weights)    
    np.save(outpath+batchname+'desired_spkslist',desired_numspkslist)
    np.save(outpath+batchname+'numspkslist',numspkslist)
    
    