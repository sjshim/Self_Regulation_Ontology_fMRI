"""
some util functions
"""
from collections import defaultdict
from glob import glob
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
    if task == 'ANT':
        contrasts = [('congruent', 'congruent'),
                     ('incongruent', 'incongruent'),
                     ('spatial', 'spatial'),
                     ('double', 'double'),
                    ('orienting_network', 'spatial-double'),
                    ('conflict_network', 'incongruent-congruent')]
    elif task == 'discountFix':
        contrasts = [('subjective_value', 'subjective_value'),
                     ('Larger-Later', 'larger_later'),
                     ('Smaller-Sooner', 'smaller_sooner'),
                     ('LL-SS', 'larger_later-smaller_sooner')]
    elif task == 'DPX':
        contrasts = [('AX', 'AX'),
                     ('BX', 'BX'),
                     ('AY', 'AY'),
                     ('BY', 'BY'),
                     ('AY-BY', 'AY-BY'),
                     ('BX-BY', 'BX-BY')]
    elif task == 'motorSelectiveStop':
        contrasts = [('crit_go', 'crit_go'),
                     ('crit_stop_success', 'crit_stop_success'),
                     ('crit_stop_failure', 'crit_stop_failure'),
                     ('noncrit_signal', 'noncrit_signal'),
                     ('noncrit_nosignal', 'noncrit_nosignal'),
                     ('crit_stop_success-crit_go', 'crit_stop_success-crit_go'),
                     ('crit_stop_failure-crit_go', 'crit_stop_failure-crit_go'),
                     ('crit_stop_success-crit_stop_failure', 'crit_stop_success-crit_stop_failure'),
                     ('crit_go-noncrit_nosignal', 'crit_go-noncrit_nosignal'),
                     ('noncrit_signal-noncrit_nosignal', 'noncrit_signal-noncrit_nosignal'),
                     ('crit_stop_success-noncrit_signal', 'crit_stop_success-noncrit_signal'),
                     ('crit_stop_failure-noncrit_signal', 'crit_stop_failure-noncrit_signal')]
    elif task == 'stroop':
        contrasts = [('congruent', 'congruent'),
                     ('incongruent', 'congruent'),
                    ('stroop', 'incongruent-congruent')]
    
    elif task == 'stopSignal':
        contrasts = [('go', 'go'),
                     ('stop_success', 'stop_success'),
                     ('stop_failure', 'stop_failure'),
                     ('stop_success-go', 'stop_success-go'),
                     ('stop_failure-go', 'stop_failure-go'),
                     ('stop_success-stop_failure', 'stop_success-stop_failure'),
                     ('stop_failure-stop_success', 'stop_failure-stop_success')]
    elif task == 'twoByTwo':
        contrasts = [('task_switch_900', 'task_switch_900'),
                     ('task_stay/cue_switch_900', 'task_stay/cue_switch_900'),
                     ('cue_stay_900', 'cue_stay_900'),
                     ('task_switch_100', 'task_switch_100'),
                     ('task_stay/cue_switch_100', 'task_stay/cue_switch_100'),
                     ('cue_stay_100', 'cue_stay_100'),
                     ('task_switch', 'task_switch_900+task_switch_100'),
                     ('task_stay/cue_switch', 'task_stay/cue_switch_900+task_stay/cue_switch_100'),
                     ('cue_stay', 'cue_stay_900+cue_stay_100'),
                     ('task_switch_cost_900', 'task_switch_900-task_stay/cue_switch_900'),
                     ('cue_switch_cost_900', 'task_stay/cue_switch_900-cue_stay_900'),
                     ('task_switch_cost_100', 'task_switch_100-task_stay/cue_switch_100'),
                     ('cue_switch_cost_100', 'task_stay/cue_switch_100-cue_stay_100'),
                     ('task_switch_cost', '(task_switch_900+task_switch_100)-(task_stay/cue_switch_900+task_stay/cue_switch_100)'),
                     ('cue_switch_cost', '(task_stay/cue_switch_900+task_stay/cue_switch_100)-(cue_stay_900+cue_stay_100)')]
        
    if regress_rt:
        contrasts.append(('RT','response_time'))
    return contrasts


def load_atlas(atlas_path, atlas_label_path=None):
    out = {}
    out['maps'] = atlas_path
    if atlas_label_path:
        file_data = np.loadtxt(atlas_label_path, 
                               dtype={'names': ('index', 'label'),
                                      'formats': ('i4', 'S50')})
        out['labels'] = [i[1].decode('UTF-8') for i in file_data]
    return out

