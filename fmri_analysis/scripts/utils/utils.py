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
def get_contrasts(task, regress_rt=True):
    if task == 'ANT':
        contrasts = [('congruent', 'congruent'),
                     ('incongruent', 'incongruent'),
                     ('spatial', 'spatial'),
                     ('double', 'double'),
                    ('orienting_network', 'spatial-double'),
                    ('conflict_network', 'incongruent-congruent')]
        
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

