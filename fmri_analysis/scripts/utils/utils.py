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
def get_flags(regress_rt=False, beta=False, cond_rt=False):
    rt_flag = "RT-True" if regress_rt else "RT-False"
    beta_flag = "beta-True" if beta else "beta-False"
    cond_rt_flag = "cond_RT-True" if cond_rt else "cond_RT-False"
    return rt_flag, beta_flag, cond_rt_flag

def get_contrasts(task, regress_rt=True, cond_rt=False):
    """ 
    Gets a list of contrasts given a task
    
    Returned list is a set of tuples, where the first element is the name
    of the contrast, and the second element is the contrast definition
    
    Args:
        task: (str) defines the task
        regress_rt: (bool) whether to include rt as a regressor 
    
    """
    contrasts=[]
    if task == 'cuedTS':
        contrasts = [('cstay', 'cstay'),
                    ('cswitch_tstay', 'cswitch_tstay'),
                    ('cswitch_tswitch', 'cswitch_tswitch'),
                    ('cue_switch_cost', 'cswitch_tstay-cstay'),
                    ('task_switch_cost', 'cswitch_tswitch-cswitch_tstay'),
                    ('task', 'cstay+cswitch_tstay+cswitch_tswitch')]
    elif task == 'directedForgetting':
        contrasts = [('con', 'con'),
                     ('pos', 'pos'),
                     ('neg', 'neg'),
                     ('neg-con', 'neg-con'),
                     ('task', 'con+pos+neg')] 
    elif task == 'flanker':
        contrasts = [('congruency_parametric', 'congruency_parametric'),
                    ('task', 'task')]
    elif task == 'goNogo':
        contrasts = [('go', 'go'),
                    ('nogo_success', 'nogo_success'),
                    ('task', 'go+nogo_success')]
    elif task == 'goNogo_nogo_failure':
        contrasts = [('go', 'go'),
                    ('nogo_failure', 'nogo_failure'),
                    ('nogo_success', 'nogo_success'),
                    ('nogo_failure-nogo_success', 'nogo_failure-nogo_success'),
                    ('nogo_failure-go', 'nogo_failure-go'),
                    ('task', 'go+nogo_failure+nogo_success')]
    elif task == 'nBack':
        contrasts = [('two_back-one_back', 'two_back-one_back'),
                    ('condition_parametric', 'condition_parametric'), 
                    ('task', 'task')]
    elif task == 'shapeMatching':
        contrasts = [('SSS', 'SSS'),
                     ('SDD', 'SDD'),
                     ('SNN', 'SNN'),
                     ('DSD', 'DSD'),
                     ('DDD', 'DDD'),
                     ('DDS', 'DDS'),
                     ('DNN', 'DNN'),
                     ('main_vars', '(SDD+DDD+DDS)-(SNN+DNN)'), 
                     ('task', 'SSS+SDD+SNN+DSD+DDD+DDS+DNN')]
    elif task == 'spatialTS':  #add non-parametric regressors 
        contrasts = [('cstay', 'tstay_cstay'),
                    ('cswitch_tstay', 'tstay_cswitch'),
                    ('cswitch_tswitch', 'tswitch_cswitch'),
                    ('cue_switch_cost', 'tstay_cswitch-tstay_cstay'),
                    ('task_switch_cost', 'tswitch_cswitch-tstay_cswitch'),
                    ('task', 'tstay_cstay+tstay_cswitch+tswitch_cswitch')]
    elif task == 'stopSignal':
        contrasts = [('go', 'go'),
                     ('stop_success', 'stop_success'),
                     ('stop_failure', 'stop_failure'),
                     ('stop_success-go', 'stop_success-go'), #for PCA
                     ('stop_failure-go', 'stop_failure-go'), #for PCA
                     ('stop_success-stop_failure', 'stop_success-stop_failure'),
                     ('stop_failure-stop_success', 'stop_failure-stop_success'),
                     ('task', 'go + stop_failure + stop_success')]
    elif task == 'stopSignalWDirectedForgetting':
        contrasts = [('go_con', 'go_con'),
                     ('go_pos', 'go_pos'),
                     ('go_neg', 'go_neg'),
                     ('stop_success_con', 'stop_success_con'),
                     ('stop_success_pos', 'stop_success_pos'),
                     ('stop_success_neg', 'stop_success_neg'),
                     ('(stop_success_con+stop_success_pos+stop_success_neg)-(go_con+go_pos_+go_neg)','(stop_success_con+stop_success_pos+stop_success_neg)-(go_con+go_pos_+go_neg)'),
                     ('(stop_failure_con+stop_failure_pos+stop_failure_neg)-(go_con+go_pos_+go_neg)','(stop_failure_con+stop_failure_pos+stop_failure_neg)-(go_con+go_pos_+go_neg)'),
                     ('(stop_success_neg-go_neg)-(stop_success_con-go_con)','(stop_success_neg-go_neg)-(stop_success_con-go_con)'),
                     ('(stop_failure_neg-go_neg)-(stop_failure_con-go_con)','(stop_failure_neg-go_neg)-(stop_failure_con-go_con)'),
                     ('stop_failure_con', 'stop_failure_con'),
                     ('stop_failure_neg', 'stop_failure_neg'),
                     ('stop_failure_pos', 'stop_failure_pos'),
                     ('task', 'go_con+go_pos+go_neg+stop_success_con+stop_success_pos+stop_success_neg+stop_failure_con+stop_failure_pos+stop_failure_neg')]
    elif task == 'stopSignalWFlanker':
        contrasts = [('go_congruent', 'go_congruent'),
                     ('go_incongruent', 'go_incongruent'),
                     ('stop_success_congruent', 'stop_success_congruent'),
                     ('stop_success_incongruent', 'stop_success_incongruent'),
                     ('(stop_success_congruent+stop_success_incongruent)-(go_congruent+go_incongruent)','(stop_success_congruent+stop_success_incongruent)-(go_congruent+go_incongruent)'),
                     ('(stop_failure_congruent+stop_failure_incongruent)-(go_congruent+go_incongruent)','(stop_failure_congruent+stop_failure_incongruent)-(go_congruent+go_incongruent)'),
                     ('(stop_success_incongruent-go_incongruent)-(stop_success_congruent-go_congruent)','(stop_success_incongruent-go_incongruent)-(stop_success_congruent-go_congruent)'),
                     ('(stop_failure_incongruent-go_incongruent)-(stop_failure_congruent-go_congruent)','(stop_failure_incongruent-go_incongruent)-(stop_failure_congruent-go_congruent)'),
                     ('stop_failure_congruent', 'stop_failure_congruent'),
                     ('stop_failure_incongruent', 'stop_failure_incongruent'),
                     ('task', 'go_congruent+go_incongruent+stop_success_congruent+stop_success_incongruent+stop_failure_congruent+stop_failure_incongruent')]
    elif task == 'directedForgettingWFlanker':
        contrasts = [('congruent_con', 'congruent_con'),
                     ('congruent_pos', 'congruent_pos'),
                     ('congruent_neg', 'congruent_neg'),
                     ('(incongruent_neg-incongruent_con)-(congruent_neg-congruent_con)','(incongruent_neg-incongruent_con)-(congruent_neg-congruent_con)'),
                     ('incongruent_con', 'incongruent_con'),
                     ('incongruent_pos', 'incongruent_pos'),
                     ('incongruent_neg', 'incongruent_neg'),
                     ('task', 'congruent_con+congruent_pos+congruent_neg+incongruent_con+incongruent_pos+incongruent_neg')]
    if regress_rt:
        contrasts.append(('RT', 'response_time'))
    
    if cond_rt:
        if task == 'cuedTS':
            contrasts.append(('cstay_RT', 'cstay_RT'))
            contrasts.append(('cswitch_tswitch_RT', 'cswitch_tswitch_RT'))
            contrasts.append(('cswitch_tstay_RT', 'cswitch_tstay_RT'))
        elif task == 'directedForgetting':
            contrasts.append(('pos_RT', 'pos_RT'))
            contrasts.append(('con_RT', 'con_RT'))
            contrasts.append(('neg_RT', 'neg_RT'))
        elif task == 'flanker':
            contrasts.append(('congruent_RT', 'congruent_RT'))
            contrasts.append(('incongruent_RT', 'incongruent_RT'))
        elif task == 'goNogo':
            contrasts.append(('go_RT', 'go_RT'))
        elif task == 'nBack':
            contrasts.append(('match_RT', 'match_RT'))
            contrasts.append(('mismatch_RT', 'mismatch_RT'))
        elif task == 'shapeMatching':
            contrasts.append(('SSS_RT', 'SSS_RT'))
            contrasts.append(('SDD_RT', 'SDD_RT'))
            contrasts.append(('SNN_RT', 'SNN_RT'))
            contrasts.append(('DSD_RT', 'DSD_RT'))
            contrasts.append(('DDD_RT', 'DDD_RT'))
            contrasts.append(('DDS_RT', 'DDS_RT'))
            contrasts.append(('DNN_RT', 'DNN_RT'))
        elif task == 'spatialTS':
            contrasts.append(('cstay_RT', 'tstay_cstay_RT'))
            contrasts.append(('cswitch_tswitch_RT', 'tswitch_cswitch_RT'))
            contrasts.append(('cswitch_tstay_RT', 'tstay_cswitch_RT'))
        elif task == 'stopSignal':
            contrasts.append(('go_RT', 'go_RT'))
            contrasts.append(('stop_failure_RT', 'stop_failure_RT'))
        elif task == 'stopSignalWDirectedForgetting':
            contrasts.append(('go_con_RT', 'go_con_RT'))
            contrasts.append(('stop_failure_con_RT', 'stop_failure_con_RT'))
            contrasts.append(('go_pos_RT', 'go_pos_RT'))
            contrasts.append(('stop_failure_pos_RT', 'stop_failure_pos_RT'))
            contrasts.append(('go_neg_RT', 'go_neg_RT'))
            contrasts.append(('stop_failure_neg_RT', 'stop_failure_neg_RT'))
        elif task == 'stopSignalWFlanker':
            contrasts.append(('go_congruent_RT', 'go_congruent_RT'))
            contrasts.append(('stop_failure_congruent_RT', 'stop_failure_congruent_RT'))
            contrasts.append(('go_incongruent_RT', 'go_incongruent_RT'))
            contrasts.append(('stop_failure_incongruent_RT', 'stop_failure_incongruent_RT'))
        elif task == 'directedForgettingWFlanker':
            contrasts.append(('congruent_pos_RT', 'congruent_pos_RT'))
            contrasts.append(('congruent_con_RT', 'congruent_con_RT'))
            contrasts.append(('congruent_neg_RT', 'congruent_neg_RT'))
            contrasts.append(('incongruent_pos_RT', 'incongruent_pos_RT'))
            contrasts.append(('incongruent_con_RT', 'incongruent_con_RT'))
            contrasts.append(('incongruent_neg_RT', 'incongruent_neg_RT'))
            
    return contrasts

def load_atlas(atlas_path, atlas_label_path=None):
    out = {'maps': atlas_path}
    if atlas_label_path:
        file_data = np.loadtxt(atlas_label_path, 
                               dtype={'names': ('index', 'label'),
                                      'formats': ('i4', 'S50')})
        out['labels'] = [i[1].decode('UTF-8') for i in file_data]
    return out

