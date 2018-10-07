# Tempytron code

## tempytron_lib.py
holds all functions for the implementation

## tempytron_validation.ipynb 
a jupyter/ipython notebook showing the validation of the model and eligibilities using a numerical implementation whose functions are explicitly defined in this file.

## corr_learn_on_task.py 
wraps together the functions in tempytron_lib.py to perform correlation-based tempotron learning. 
It takes as input arguments a sequence of integers as the feature labels and it outputs weight dynamics and learning curve data.
The end of the jupyter notebook runs this, reads in its output, and plots it.

## Tempytron_correlations.nb
A mathematica notebook that computes the simple, but somewhat messy correlation functions used in the get_inspk_eligibilities function.
