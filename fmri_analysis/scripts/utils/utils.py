"""
some util functions
"""
from collections import defaultdict
import numpy as np
from os.path import join, sep
import pandas as pd


# ********************************************************
# Basic Help Methods
# ********************************************************
def get_flags(regress_rt=False, beta=False):
    rt_flag = "RT-True" if regress_rt else "RT-False"
    beta_flag = "beta-True" if beta else "beta-False"
    return rt_flag, beta_flag

def get_contrasts(task, regress_rt=True):
    """ 
    Gets a list of contrasts given a task
    
    Returned list is a set of tuples, where the first element is the name
    of the contrast, and the second element is the contrast definition
    
    Args:
        task: (str) defines the task
        regress_rt: (bool) whether to include rt as a regressor 
    
    """
    contrasts=[]
    if task == 'flanker':
        contrasts = [('congruency_parametric', 'congruency_parametric'), #use for PCA
                    ('task', 'task')]
    elif task == 'goNogo':
        contrasts = [('congruency_parametric', 'congruency_parametric'), #use for PCA
                    ('task', 'task')]
    elif task == 'nBack':
        contrasts = [('congruency_parametric', 'congruency_parametric'), #use for PCA
                    ('task', 'task')]
    elif task == 'directedForgetting':
        contrasts = [('con', 'con'),
                     ('pos', 'pos'),
                     ('neg', 'neg'),
                     ('task', 'task')] 
    elif task == 'motorSelectiveStop':
        contrasts = [('crit_go', 'crit_go'),
                     ('crit_stop_success', 'crit_stop_success'),
                     ('crit_stop_failure', 'crit_stop_failure'),
                     ('noncrit_signal', 'noncrit_signal'),
                     ('noncrit_nosignal', 'noncrit_nosignal'),
                     ('crit_stop_success-crit_go', 'crit_stop_success-crit_go'), #for PCA
                     ('crit_stop_failure-crit_go', 'crit_stop_failure-crit_go'), #for PCA
                     ('crit_stop_success-crit_stop_failure', 'crit_stop_success-crit_stop_failure'),
                     ('crit_go-noncrit_nosignal', 'crit_go-noncrit_nosignal'),
                     ('noncrit_signal-noncrit_nosignal', 'noncrit_signal-noncrit_nosignal'),
                     ('crit_stop_success-noncrit_signal', 'crit_stop_success-noncrit_signal'),
                     ('crit_stop_failure-noncrit_signal', 'crit_stop_failure-noncrit_signal'),
                     ('task', 'crit_go + crit_stop_success + crit_stop_failure + noncrit_signal + noncrit_nosignal')]
    elif task == 'manipulationTask':  #add non-parametric regressors 
        contrasts =  [('cue', 'cue'),
                     ('probe', 'probe'),
                     ('rating', 'rating')]
    elif task == 'stroop':
        contrasts = [('congruency', 'congruency_parametric'), #for PCA
                     ('task', 'task')]
    elif task == 'stopSignal':
        contrasts = [('go', 'go'),
                     ('stop_success', 'stop_success'),
                     ('stop_failure', 'stop_failure'),
                     ('stop_success-go', 'stop_success-go'), #for PCA
                     ('stop_failure-go', 'stop_failure-go'), #for PCA
                     ('stop_success-stop_failure', 'stop_success-stop_failure'),
                     ('stop_failure-stop_success', 'stop_failure-stop_success'),
                     ('task', 'go + stop_failure + stop_success')]
    elif task == 'twoByTwo':
        contrasts = [('task_switch_cost_900', 'task_switch_900-task_stay_cue_switch_900'),
                     ('cue_switch_cost_900', 'task_stay_cue_switch_900-cue_stay_900'),
                     ('task_switch_cost_100', 'task_switch_100-task_stay_cue_switch_100'),
                     ('cue_switch_cost_100', 'task_stay_cue_switch_100-cue_stay_100'),
                     ('task_switch_cost', '(task_switch_900+task_switch_100)-(task_stay_cue_switch_900+task_stay_cue_switch_100)'), #for PCA
                     ('cue_switch_cost', '(task_stay_cue_switch_900+task_stay_cue_switch_100)-(cue_stay_900+cue_stay_100)'), #for PCA
                     ('task', 'task_switch_900 + task_switch_100 + task_stay_cue_switch_900 + task_stay_cue_switch_100 + cue_stay_900 + cue_stay_100')]
        for trial in ['task_switch', 'task_stay_cue_switch', 'cue_stay']: #add regressor for each trial type
            contrasts.append((trial, '%s_900 + %s_100' % (trial, trial)))
            for CSI in ['100', '900']:
                contrasts.append((trial+'_'+CSI, trial+'_'+CSI))
    elif task == 'WATT3': #all regressors save RT
        contrasts = [('practice', 'practice'),
                    ('planning_event','planning_event'),
                    ('planning_parametric','planning_parametric'), #for PCA
                    ('acting_event','acting_event'),
                    ('acting_parametric', 'acting_parametric'),
                    ('feedback','feedback'),
                    ('task', 'planning_event + acting_event'),
                    ('task_parametric', 'planning_parametric + acting_parametric')]
    if regress_rt:
        if task == 'motorSelectiveStop':
            contrasts.append(('crit_go_RT', 'crit_go_RT'))
            contrasts.append(('noncrit_signal_RT', 'noncrit_signal_RT'))
            contrasts.append(('noncrit_nosignal_RT', 'noncrit_signal_RT'))
            contrasts.append(('crit_stop_failure_RT', 'crit_stop_failure_RT'))
        elif task == 'stopSignal':
            contrasts.append(('go_RT', 'go_RT'))
            contrasts.append(('stop_failure_RT', 'stop_failure_RT'))
        elif task== 'WATT3':
            contrasts.append(('planning_RT', 'planning_RT'))
            contrasts.append(('acting_RT', 'acting_RT'))
        elif task == 'CCTHot':
            contrasts.append(('first_RT', 'first_RT'))
            contrasts.append(('subsequent_RT', 'subsequent_RT'))
        elif task == 'DPX':
            contrasts.append(('AX_RT', 'AX_RT'))
            contrasts.append(('AY_RT', 'AY_RT'))
            contrasts.append(('BX_RT', 'BX_RT'))
            contrasts.append(('BY_RT', 'BY_RT'))
        else:         
            contrasts.append(('RT','response_time'))
    return contrasts

def load_atlas(atlas_path, atlas_label_path=None):
    out = {'maps': atlas_path}
    if atlas_label_path:
        file_data = np.loadtxt(atlas_label_path, 
                               dtype={'names': ('index', 'label'),
                                      'formats': ('i4', 'S50')})
        out['labels'] = [i[1].decode('UTF-8') for i in file_data]
    return out

