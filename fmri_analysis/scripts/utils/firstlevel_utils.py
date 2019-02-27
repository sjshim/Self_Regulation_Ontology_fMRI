from collections import namedtuple
from glob import glob
from nistats.design_matrix import make_first_level_design_matrix
import numpy as np
from os import makedirs, path
import pandas as pd
import pickle
import random
from sklearn.preprocessing import scale
from utils.events_utils import get_beta_series, parse_EVs
from utils.utils import get_contrasts

# ********************************************************
# helper functions 
# ******************************************************** 

def temp_deriv(dataframe, columns=None):
    if columns is None:
        columns = dataframe.columns
    td = dataframe.loc[:,columns].apply(np.gradient)
    td.iloc[0,:] = 0
    for i,col in td.iteritems():
        insert_loc = dataframe.columns.get_loc(i)
        dataframe.insert(insert_loc+1, i+'_TD', col)   

def create_design(events, confounds, task, TR, beta=True, regress_rt=False):
    if beta:
        EV_dict = get_beta_series(events, regress_rt=regress_rt)
    else:
        EV_dict = parse_EVs(events, task, regress_rt=regress_rt)
    paradigm = get_paradigm(EV_dict)
    # make design
    n_scans = int(confounds.shape[0])
    design = make_first_level_design_matrix(np.arange(n_scans)*TR,
                               paradigm,
                               drift_model='cosine',
                               add_regs=confounds.values,
                               add_reg_names=list(confounds.columns))
    design = design.apply(scale)
    # add temporal derivative to task columns
    task_cols = [i for i in paradigm.trial_type.unique() if i != 'junk']
    temp_deriv(design, task_cols)
    return design

def make_first_level_obj(subject_id, task, fmriprep_dir, data_dir, TR, 
                        regress_rt=False, beta=False):
    func_file, mask_file = get_func_file(fmriprep_dir, subject_id, task)
    if func_file is None or mask_file is None:
        print("Missing MRI files for %s: %s" % (subject_id, task))
        return None
    events = get_events(data_dir, subject_id, task)
    if events is None:
        print("Missing event files for %s: %s" % (subject_id, task))
        return None
    confounds = get_confounds(fmriprep_dir, subject_id, task)
    design = create_design(events, confounds, task, TR, beta=beta, regress_rt=regress_rt)
    contrasts = get_contrasts(task, design)
    subjinfo = FirstLevel(func_file, mask_file, design, contrasts, '%s_%s' % (subject_id, task))
    subjinfo.events = events
    subjinfo.model_settings['beta'] = beta
    subjinfo.model_settings['regress_rt'] = regress_rt
    return subjinfo

def save_first_level_obj(subjinfo, output_dir):
    subj, task = subjinfo.ID.split('_')
    directory = path.join(output_dir, subj, task)
    rt_flag = "True" if subjinfo.model_settings['regress_rt'] else "False"
    beta_flag = "True" if subjinfo.model_settings['beta'] else "False"
    filename = path.join(directory, 'firstlevel_RT-%s_beta-%s.pkl' % (rt_flag, beta_flag))
    makedirs(directory, exist_ok=True)
    f = open(filename, 'wb')
    pickle.dump(subjinfo, f)
    f.close()

def get_first_level_objs(subject_id, task, first_level_dir, regress_rt=False, beta=False):
    rt_flag = 'RT-True' if regress_rt else 'RT-False'
    beta_flag = 'beta-True' if beta else 'beta-False'
    files = path.join(first_level_dir, subject_id, task, '*%s_%s*pkl' % (rt_flag, beta_flag))
    return glob(files)    

def load_first_level_objs(task, output_dir, regress_rt=False, beta=False):
    subjinfos = []
    files = get_first_level_objs(*, task, output_dir, 
                                     regress_rt=regress_rt, beta=beta)
    for filey in files:
        f = open(filey, 'rb')
        subjinfos.append(pickle.load(f))
        f.close()
    return subjinfos



# ********************************************************
# helper classes 
# ******************************************************** 

SubjInfo = namedtuple('subjinfo', ['func','mask','design','contrasts','ID'])
class FirstLevel():
    def __init__(self, func, mask, design, contrasts, ID):
        self.func = func
        self.mask = mask
        self.design = design
        self.contrasts = contrasts
        self.ID = ID
        # for model
        self.model_settings = {'beta': False, 'regress_rt': False}
        self.fit_model = None
        self.maps = {}
    
    def get_subjinfo(self):
        return SubjInfo(self.func, 
                        self.mask, 
                        self.design, 
                        self.contrasts, 
                        self.ID)
    
    def __str__(self):
        s = """
            ** %s **
                * Func File: %s
                * Mask File: %s
                * Model Settings: %s
            """ % (self.ID, self.func, self.mask, self.model_settings)
        return s
    
# ********************************************************
# Process Functions
# ******************************************************** 
def process_confounds(confounds_file):
    """
    scrubbing for TASK
    remove TRs where FD>.5, stdDVARS (that relates to DVARS>.5)
    regressors to use
    ['X','Y','Z','RotX','RotY','RotY','<-firsttemporalderivative','stdDVARs','FD','respiratory','physio','aCompCor0-5']
    junk regressor: errors, ommissions, maybe very fast RTs (less than 50 ms)
    """
    confounds_df = pd.read_csv(confounds_file, sep = '\t', 
                               na_values=['n/a']).fillna(0)
    excessive_movement = (confounds_df.FramewiseDisplacement>.5) & \
                            (confounds_df.stdDVARS>1.2)
    excessive_movement_TRs = excessive_movement[excessive_movement].index
    excessive_movement_regressors = np.zeros([confounds_df.shape[0], 
                                   np.sum(excessive_movement)])
    for i,TR in enumerate(excessive_movement_TRs):
        excessive_movement_regressors[TR,i] = 1
    excessive_movement_regressor_names = ['rejectTR_%d' % TR for TR in 
                                          excessive_movement_TRs]
    # get movement regressors
    movement_regressor_names = ['X','Y','Z','RotX','RotY','RotZ']
    movement_regressors = confounds_df.loc[:,movement_regressor_names]
    movement_regressor_names += ['Xtd','Ytd','Ztd','RotXtd','RotYtd','RotZtd']
    movement_regressors = np.hstack((movement_regressors, np.gradient(movement_regressors,axis=0)))
    # add square
    movement_regressor_names += [i+'_sq' for i in movement_regressor_names]
    movement_regressors = np.hstack((movement_regressors, movement_regressors**2))
    
    # add additional relevant regressors
    add_regressor_names = ['FramewiseDisplacement'] 
    #add_regressor_names += [i for i in confounds_df.columns if 'aCompCor' in i]
    additional_regressors = confounds_df.loc[:,add_regressor_names].values
    regressors = np.hstack((movement_regressors,
                            additional_regressors,
                            excessive_movement_regressors))
    # concatenate regressor names
    regressor_names = movement_regressor_names + add_regressor_names + \
                      excessive_movement_regressor_names
    return regressors, regressor_names
        
def process_physio(cardiac_file, resp_file):
    cardiac_file = '/mnt/temp/sub-s130/ses-1/func/sub-s130_ses-1_task-stroop_run-1_recording-cardiac_physio.tsv.gz'
    resp_file = '/mnt/temp/sub-s130/ses-1/func/sub-s130_ses-1_task-stroop_run-1_recording-respiratory_physio.tsv.gz'
    
    
# ********************************************************
# Getter functions
# ******************************************************** 

def get_func_file(fmriprep_dir, subject_id, task):
    # strip "sub" from beginning of subject_id if provided
    subject_id = subject_id.replace('sub-','')
    
    # get mask_file
    func_file = glob(path.join(fmriprep_dir,
                          'sub-%s' % subject_id,
                          '*', 'func',
                          '*%s*MNI*preproc.nii.gz' % task))
    
    # get mask_file
    mask_file = glob(path.join(fmriprep_dir,
                          'sub-%s' % subject_id,
                          '*', 'func',
                          '*%s*MNI*brainmask.nii.gz' % task))
    if not func_file or not mask_file:
        return None, None
    return func_file[0], mask_file[0]

def get_confounds(fmriprep_dir, subject_id, task):
    # strip "sub" from beginning of subject_id if provided
    subject_id = subject_id.replace('sub-','')
    
    ## Get the Confounds File (output of fmriprep)
    # Read the TSV file and convert to pandas dataframe
    confounds_file = glob(path.join(fmriprep_dir,
                               'sub-%s' % subject_id,
                               '*', 'func',
                               '*%s*confounds.tsv' % task))[0]
    regressors, regressor_names = process_confounds(confounds_file)
    confounds = pd.DataFrame(regressors, columns=regressor_names)
    return confounds
    
def get_events(data_dir, subject_id, task):
    ## Get the Events File if it exists
    # Read the TSV file and convert to pandas dataframe
    try:
        event_file = glob(path.join(data_dir,
                               'sub-%s' % subject_id,
                               '*', 'func',
                               '*%s*events.tsv' % task))[0]   
        events_df = pd.read_csv(event_file,sep = '\t')
        return events_df
    except IndexError:
        return None

def get_paradigm(EV_dict):
    # convert nipype format to nistats paradigm
    conditions = []
    onsets = []
    amplitudes = []
    durations = []
    for i in range(len(EV_dict['conditions'])):
        onset = EV_dict['onsets'][i]
        onsets += onset
        # add on conditions
        conditions += [EV_dict['conditions'][i]]*len(onset)
        duration = EV_dict['durations'][i]
        # add on duration, and extend if the array has a length of one
        if len(duration) == 1 and len(onset) > 1:
            duration *= len(onset)
        durations += duration
        # add on amplitude, and extend if the array has a length of one
        amplitude = EV_dict['amplitudes'][i]
        if len(amplitude) == 1 and len(onset) > 1:
            amplitude *= len(onset)
        amplitudes += amplitude
    paradigm = {'trial_type': conditions,
               'onset': onsets,
               'modulation': amplitudes,
               'duration': durations} 
    paradigm = pd.DataFrame(paradigm).sort_values(by='onset').reset_index()
    return paradigm
