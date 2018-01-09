"""
some util functions
"""
from collections import OrderedDict as odict
from glob import glob
import nilearn
from nilearn import image, input_data
import numpy as np
from os.path import basename, dirname, join, exists
import pandas as pd
import pickle
import re
import shutil
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

# ********************************************************
# Behavioral Utility Functions
# ********************************************************

def move_EV(subj, task, events_dir, fmri_dir):
    subj = subj.replace('sub-','')
    # get event file
    ev_file = glob(join(events_dir,'*%s*%s*' % (subj, task)))[0]
    task_fmri_files = glob(join(fmri_dir, '*%s*' % subj,'*', 
                                'func','*%s*bold*' % task))
    task_fmri_dir = dirname(task_fmri_files[0])
    base_name = basename(task_fmri_files[0]).split('_bold')[0]
    new_events_file = join(task_fmri_dir, base_name+'_events.tsv')
    shutil.copyfile(ev_file, new_events_file)
    return new_events_file
    
def move_EVs(events_dir, fmri_dir, tasks, overwrite=True, verbose=False):
    created_files = []
    for subj_file in sorted(glob(join(fmri_dir,'sub-s???'))):
        subj = basename(subj_file)
        for task in tasks:
            if overwrite==True or not exists(join(subj_file,'*',
                                                 'func', '*%s*' % task)):
                try:
                    name = move_EV(subj, task, events_dir, fmri_dir)
                    created_files.append(name)
                except IndexError:
                    print('Move_EV failed for the %s: %s' % (subj, task))
    if verbose:
        print('\n'.join(created_files))
        
# ********************************************************
# 1st level analysis utility functions
# ********************************************************

def get_contrasts(task, regress_rt=True):
    contrast_list = []
    if task == 'ANT':
        # contrasts vs baseline
        c1 = ['incongruent','T', ['incongruent'], [1]]
        c2 = ['congruent','T', ['congruent'], [1]]
        c3 = ['spatial_cue','T', ['spatial_cue'], [1]]
        c4 = ['double_cue','T', ['double_cue'], [1]]
        # contrasts 
        c5 = ['conflict_network','T', ['incongruent','congruent'], [1,-1]]
        c6 = ['orienting_network','T', ['spatial_cue','double_cue'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6]
        if regress_rt:
            c7 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c7)
    elif task == 'CCTHot':
        # contrasts vs baseline
        c1 = ['EV','T', ['EV'], [1]]
        c2 = ['risk','T', ['risk'], [1]]
        contrast_list = [c1,c2]
        if regress_rt:
            c3 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c3)
    elif task == 'discountFix':
        # contrasts vs baseline
        c1 = ['subjective_value','T', ['subjective_value'], [1]]
        c2 = ['larger_later','T', ['larger_later'], [1]]
        c3 = ['smaller_sooner','T', ['smaller_sooner'], [1]]
        # contrasts
        c4 = ['LL_vs_SS','T', ['larger_later','smaller_sooner'], [1,-1]]
        contrast_list = [c1,c2,c3,c4]
        if regress_rt:
            c5 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c5)
    elif task == 'DPX':
        # contrasts vs baseline
        c1 = ['AX','T', ['AX'], [1]]
        c2 = ['AY','T', ['AY'], [1]]
        c3 = ['BX','T', ['BX'], [1]]
        c4 = ['BY','T', ['BY'], [1]]
        # contrasts 
        c5 = ['BX-BY','T', ['BX','BY'], [1,-1]]
        c6 = ['AY-BY','T', ['AY','BY'], [1,-1]]
        c7 = ['AY-BX','T', ['AY','BX'], [1,-1]]
        c8 = ['BX-AY','T', ['BX','AY'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8]
        if regress_rt:
            c9 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c9)
    elif task == 'motorSelectiveStop':
        # contrasts vs baseline
        c1 = ['crit_go','T', ['crit_go'], [1]]
        c2 = ['crit_stop_success','T', ['crit_stop_success'], [1]]
        c3 = ['crit_stop_failure','T', ['crit_stop_failure'], [1]]
        c4 = ['noncrit_signal','T', ['noncrit_signal'], [1]]
        c5 = ['noncrit_nosignal','T', ['noncrit_nosignal'], [1]]
        # contrasts
        c6 = ['crit_stop_success-crit_go', 'T', 
                ['crit_stop_success', 'crit_go'], [1,-1]]
        c7 = ['crit_stop_failure-crit_go', 'T', 
                ['crit_stop_failure', 'crit_go'], [1,-1]]
        c8 = ['crit_go-noncrit_nosignal', 'T', 
                ['crit_go', 'noncrit_nosignal'], [1,-1]]
        c9 = ['noncrit_signal-noncrit_nosignal', 'T' ,
                ['noncrit_signal','noncrit_nosignal'], [1,-1]]
        c10 = ['crit_stop_success-crit_stop_failure','T',
                ['crit_stop_success', 'crit_stop_failure'], [1,-1]]
        c11 = ['crit_stop_failure-crit_stop_success','T', 
                ['crit_stop_failure', 'crit_stop_success'], [1,-1]]
        c12 = ['crit_stop_success-noncrit_signal','T',
                ['crit_stop_success', 'noncrit_signal'], [1,-1]]
        c13 = ['crit_stop_failure-noncrit_signal','T',
                ['crit_stop_failure', 'noncrit_signal'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13]
    elif task == 'stopSignal':
        # contrasts vs baseline
        c1 = ['go','T', ['go'], [1]]
        c2 = ['stop_success','T', ['stop_success'], [1]]
        c3 = ['stop_failure','T', ['stop_failure'], [1]]
        # contrasts
        c4 = ['stop_success-go','T', ['stop_success', 'go'], [1,-1]]
        c5 = ['stop_failure-go','T', ['stop_failure', 'go'], [1,-1]]
        c6 = ['stop_success-stop_failure','T', 
                ['stop_success', 'stop_failure'], [1,-1]]
        c7 = ['stop_failure-stop_success','T', 
                ['stop_failure', 'stop_success'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7]
    elif task == 'stroop':
        # contrasts vs baseline
        c1 = ['incongruent','T', ['incongruent'], [1]]
        c2 = ['congruent','T', ['congruent'], [1]]
        # contrasts
        c3 = ['incongruent-congruent','T', ['incongruent','congruent'], [1,-1]]
        contrast_list = [c1,c2,c3]
        if regress_rt:
            c4 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c4)
    elif task == 'surveyMedley':
        # contrasts vs baseline
        c1 = ['stim_duration','T', ['stim_duration'], [1]]
        c2 = ['movement','T', ['movement'], [1]]
        contrast_list = [c1,c2]
        if regress_rt:
            c4 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c4)
    elif task == 'twoByTwo':
        # contrasts vs baseline
        c1 = ['cue_switch_100','T', ['cue_switch_100'], [1]]
        c2 = ['cue_stay_100','T', ['cue_stay_100'], [1]]
        c3 = ['task_switch_100','T', ['task_switch_100'], [1]]
        c4 = ['cue_switch_900','T', ['cue_switch_900'], [1]]
        c5 = ['cue_stay_900','T', ['cue_stay_900'], [1]]
        c6 = ['task_switch_900','T', ['task_switch_900'], [1]]
        # contrasts
        c7 = ['cue_switch_cost_100','T', 
                ['cue_switch_100','cue_stay_100'], [1,-1]]
        c8 = ['cue_switch_cost_900','T', 
                ['cue_switch_900','cue_stay_900'], [1,-1]]
        c9 = ['task_switch_cost_100','T', 
                ['task_switch_100','cue_switch_100'], [1,-1]]
        c10 = ['task_switch_cost_900','T', 
                ['task_switch_900','cue_switch_900'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
        if regress_rt:
            c11 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c11)
    elif task == 'WATT3':
        # contrasts vs baseline
        c1 = ['plan_PA_with','T', ['plan_PA_with'], [1]]
        c2 = ['plan_PA_without','T', ['plan_PA_without'], [1]]
        # contrasts
        c3 = ['search_depth','T', ['plan_PA_with','plan_PA_without'], [1,-1]]
        contrast_list = [c1,c2,c3]
    return contrast_list
        
# functions to extract fmri events
      
def get_ev_vars(output_dict, events_df, condition_spec, col=None, 
                amplitude=1, duration=0, subset=None, onset_column='onset'):
    """ adds amplitudes, conditions, durations and onsets to an output_dict
    
    Args:
        events_df: events file to parse
        condition_spec: string specfying condition name, or list of tuples of the fomr
            (subset_key, name) where subset_key groups the rows in col. If a list,
            col must be specified
        col: the column to be subset by the keys in conditions
        amplitude: either an int or string. If int, sets a constant amplitude. If
            string, amplitude is set to that column
        duration: either an int or string. If int, sets a constant duration. If
            string, duration is set to that column
        subset: pandas query string to subset the data before use
        onset_column: the column of timing to be used for onsets
    
    """
    required_keys =  set(['amplitudes','conditions','durations','onsets'])
    assert set(output_dict.keys()) == required_keys
    amplitudes = output_dict['amplitudes']
    conditions = output_dict['conditions']
    durations = output_dict['durations']
    onsets = output_dict['onsets']
    
    # if subset is specified as a string, use to query
    if subset is not None:
        events_df = events_df.query(subset)
    # if a column is specified, group by the values in that column
    if type(condition_spec) == list:
        assert (col is not None), "Must specify column when condition_spec is a list"
        group_df = events_df.groupby(col)
        for condition, condition_name in condition_spec:
            if type(condition) is not list:
                condition = [condition]
            # get members of group identified by the condition list
            c_dfs = [group_df.get_group(c) for c in condition 
                     if c in group_df.groups.keys()]
            if len(c_dfs)!=0:
                c_df = pd.concat(c_dfs)
                conditions.append(condition_name)
                onsets.append(c_df.loc[:,onset_column].tolist())
                if type(amplitude) in (int,float):
                    amplitudes.append([amplitude])
                elif type(amplitude) == str:
                    amplitudes.append(c_df.loc[:,amplitude].tolist())
                if type(duration) in (int,float):
                    durations.append([duration])
                elif type(duration) == str:
                    durations.append(c_df.loc[:,duration].tolist())
    elif type(condition_spec) == str:
        group_df = events_df
        conditions.append(condition_spec)
        onsets.append(group_df.loc[:,onset_column].tolist())
        if type(amplitude) in (int,float):
            amplitudes.append([amplitude])
        elif type(amplitude) == str:
            amplitudes.append(group_df.loc[:,amplitude].tolist())
        if type(duration) in (int,float):
            durations.append([duration])
        elif type(duration) == str:
            durations.append(group_df.loc[:,duration].tolist())

# specific task functions
def get_ANT_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # cue type
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('spatial','spatial_cue'), ('double', 'double_cue')],
                col='cue', 
                duration='duration')
    # conflict type
    get_ev_vars(output_dict, events_df,
                condition_spec=[('congruent','congruent'), ('incongruent', 'incongruent')],
                col='flanker_type', 
                duration='duration')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

def get_CCTHot_EVs(events_df, regress_rt):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # add main parametric regressors: EV and risk
    get_ev_vars(output_dict, events_df, 
                condition_spec='EV', 
                duration='duration', 
                amplitude='EV')
    get_ev_vars(output_dict, events_df, 
                condition_spec='risk',
                duration='duration', 
                amplitude='risk')
    # other regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec='num_click_in_round', 
                duration='duration', 
                amplitude='num_click_in_round')
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(1,'reward'), (0,'punishment')], 
                col='feedback')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

def get_discountFix_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # regressors of interest
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('larger_later','larger_later'), ('smaller_sooner','smaller_sooner')],
                col='choice', 
                duration='duration')
    get_ev_vars(output_dict, events_df, 
                condition_spec='subjective_value', 
                duration='duration', 
                amplitude='subjective_value')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')],
                col='junk', 
                duration='duration')
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

def get_DPX_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('AX','AX'), ('AY','AY'), ('BX', 'BX'), ('BY','BY')],
                col='condition', 
                duration='duration')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')  
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

def get_motorSelectiveStop_EVs(events_df):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('crit_go','crit_go'), 
                            ('crit_stop_success', 'crit_stop_success'), 
                            ('crit_stop_failure', 'crit_stop_failure'),
                            ('noncrit_signal', 'noncrit_signal'),
                            ('noncrit_nosignal', 'noncrit_nosignal')],
                col='trial_type', 
                duration='duration')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')    
    return output_dict

def get_stopSignal_EVs(events_df):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('go','go'), 
                            ('stop_success', 'stop_success'), 
                            ('stop_failure', 'stop_failure')],
                col='trial_type', 
                duration='duration')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')    
    return output_dict

def get_stroop_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('congruent','congruent'), ('incongruent','incongruent')],
                col='trial_type', 
                duration='duration')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration') 
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

def get_surveyMedley_EVs(events_df, regress_rt=True):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec='stim_duration', 
                duration='stim_duration')
    get_ev_vars(output_dict, events_df, 
                condition_spec='movement',  
                onset_column='movement_onset')
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

    
def get_twoByTwo_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # cue switch contrasts
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('switch','cue_switch_900'), ('stay','cue_stay_900')],
                col='cue_switch', 
                duration='duration',
                subset="CTI==900")
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('switch','cue_switch_100'), ('stay','cue_stay_100')],
                col='cue_switch', 
                duration='duration',
                subset="CTI==100")
    # task switch contrasts
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('switch','task_switch_900')],
                col='task_switch',
                duration='duration',
                subset="CTI==900")
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('switch','task_switch_100')],
                col='task_switch', 
                duration='duration',
                subset="CTI==100")
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time')
    return output_dict

def get_WATT3_EVs(events_df):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # planning conditions
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('UA_with_intermediate','plan_UA_with'), 
                            ('UA_without_intermediate','plan_UA_without'),
                            ('PA_with_intermediate','plan_PA_with'),
                            ('PA_without_intermediate','plan_PA_without')],
                col='condition', 
                duration='duration', 
                subset="planning==1")
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec='movement', 
                onset_column='movement_onset')
    return output_dict


# How to model RT
# For each condition model responses with constant duration 
# (average RT across subjects or block duration)
# RT as a separate regressor for each onset, constant duration, 
# amplitude as parameteric regressor (function of RT)
def parse_EVs(events_df, task, regress_rt=True):
    if task == "ANT":
        EV_dict = get_ANT_EVs(events_df, regress_rt=True)
    elif task == "CCTHot": 
        EV_dict = get_CCTHot_EVs(events_df, regress_rt=True)
    elif task == "discountFix": 
        EV_dict = get_discountFix_EVs(events_df, regress_rt=True)
    elif task == "DPX":
        EV_dict = get_DPX_EVs(events_df, regress_rt=True)
    elif task == "motorSelectiveStop": 
        EV_dict = get_motorSelectiveStop_EVs(events_df)
    elif task == 'surveyMedley':
        EV_dict = get_surveyMedley_EVs(events_df)
    elif task == "stopSignal":
        EV_dict = get_stopSignal_EVs(events_df)
    elif task == "stroop":
        EV_dict = get_stroop_EVs(events_df, regress_rt=True)
    elif task == "twoByTwo":
        EV_dict = get_twoByTwo_EVs(events_df, regress_rt=True)
    elif task == "WATT3":
        EV_dict = get_WATT3_EVs(events_df)
    return EV_dict

    
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
    movement_deriv_regressors = np.gradient(movement_regressors,axis=0)
    # add additional relevant regressors
    add_regressor_names = ['FramewiseDisplacement', 'stdDVARS'] 
    add_regressor_names += [i for i in confounds_df.columns if 'aCompCor' in i]
    additional_regressors = confounds_df.loc[:,add_regressor_names].values
    regressors = np.hstack((movement_regressors,
                            movement_deriv_regressors,
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
# 2nd level analysis utility functions
# ********************************************************

def concat_and_smooth(map_files, smoothness=None):
    """
    Loads and smooths files specified in 
    map_files and creates a dictionary of them
    """
    smooth_copes = odict()
    for img_i, img in enumerate(sorted(map_files)):
        subj = re.search('s[0-9][0-9][0-9]',img).group(0)
        smooth_cope = nilearn.image.smooth_img(img, smoothness)
        smooth_copes[subj] = smooth_cope
    return smooth_copes

def get_contrast_names(contrast_path):
    contrasts = pickle.load(open(contrast_path,'rb'))
    contrast_names = [c[0] for c in contrasts]
    return contrast_names

# function to get TS within labels
def project_contrast(img_files, parcellation_file, mask_file):
    parcellation = image.load_img(parcellation_file)
    resampled_images = image.resample_img(img_files, parcellation.affine)
    if len(parcellation.shape) == 3:
        masker = input_data.NiftiLabelsMasker(labels_img=parcellation_file, 
                                               resampling_target="labels", 
                                               standardize=False,
                                               memory='nilearn_cache', 
                                               memory_level=1)
    elif len(parcellation.shape) == 4:
         masker = input_data.NiftiMapsMasker(maps_img=parcellation_file, 
                                             mask_img=mask_file,
                                             resampling_target="maps", 
                                             standardize=False,
                                             memory='nilearn_cache',
                                             memory_level=1)
    time_series = masker.fit_transform(resampled_images)
    return time_series, masker

def create_projection_df(parcellation_file, mask_file, 
                         data_dir, tasks, filename=None):
    
    # project contrasts into lower dimensional space    
    projections = []
    index = []
    for task in tasks:
        # get all contrasts
        contrast_path = glob(join(data_dir,'*%s/contrasts.pkl' % task))
        if len(contrast_path)>0:
            contrast_path = contrast_path[0]
        else:
            continue # move to next iteration if no contrast files found
        contrast_names = get_contrast_names(contrast_path)
        # for each contrast, project into space defined by parcellation file
        for i,name in enumerate(contrast_names):
            func_files = sorted(glob(join(data_dir, '*%s/zstat%s.nii.gz' 
                                          % (task, i+1))))
            TS, masker = project_contrast(func_files,
                                          parcellation_file, 
                                          mask_file)
            projections.append(TS)
            index += [re.search('s[0-9][0-9][0-9]',f).group(0)
                        + '_%s_%s' % (task, name)
                        for f in func_files]
    projections_df = pd.DataFrame(np.vstack(projections), index)
    
    # split index into column names
    subj = [i[:4] for i in projections_df.index]
    contrast = [i[5:] for i in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(0, 'contrast', contrast)
    
    # save
    if filename:
        projections_df.to_json(filename)
    return projections_df

# functions on projections df
def create_neural_feature_mat(projections_df, filename=None):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    subj = [i[:4] for i in projections_df.index]
    contrast = [i[5:] for i in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(0, 'contrast', contrast)
    neural_feature_mat = projections_df.pivot(index='subj', columns='contrast')
    if filename:
        neural_feature_mat.to_json(filename)
    return neural_feature_mat

def projections_corr(projections_df, remove_global=True, grouping=None):
    """ Create a correlation matrix of a projections dataframe
    
    Args:
        projections_df: a projection_df, as create by create_projection_df
        remove_global: if True, subtract the mean contrast
        grouping: "subj" or "contrast". If provided, average over the group
        
    Returns:
        Correlation Matrix
    """
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert exists(projections_df)
        projections_df = pd.read_json(projections_df)
    
    if remove_global:
        projections_df.iloc[:,2:] -= projections_df.mean()
    if grouping:
        projections_df = projections_df.groupby(grouping).mean()
    return projections_df.T.corr()



def get_confusion_matrix(projections_df, normalize=True):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    X = projections_df.iloc[:, 2:]
    y = projections_df.contrast
    clf = LogisticRegressionCV(multi_class='multinomial')
    predict = cross_val_predict(clf, X, y, cv=10)
    cm = confusion_matrix(y, predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
    
                                           
