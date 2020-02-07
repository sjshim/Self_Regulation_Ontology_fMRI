"""
some util functions
"""
import numpy as np
import pandas as pd
import pdb

# ********************************************************
# helper_functions
# ********************************************************  
def normalize_rt(events_df, groupby=None):  
    """demeans RT by condition"""
    if groupby is None:
        rt = events_df.response_time
        events_df.loc[:,'response_time'] = rt - rt[rt>0].mean()
    else:
        for name, group in events_df.groupby(groupby):
            rt = group.response_time
            events_df.loc[group.index,'response_time'] = rt - rt[rt>0].mean()
    
# ********************************************************
# 1st level analysis utility functions
# ********************************************************        
# functions to extract fmri events        
def get_ev_vars(output_dict, events_df, condition_spec, col=None, 
                amplitude=1, duration=0, subset=None, onset_column='onset'):
    """ adds amplitudes, conditions, durations and onsets to an output_dict
    
    Args:
        events_df: events file to parse
        condition_spec: string specfying condition name, or list of tuples of the form (subset_key, name) where subset_key are one or more groups in col. If a list, col must be specified. 
        col: the column to be subset by the keys in conditions
        amplitude: either an int or string. If int, sets a constant amplitude. If string, ... 
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
    # if amplitudes or durations were passed as a series, subset and convert to list
    if type(duration) == pd.core.series.Series:
        duration = duration[events_df.index].tolist()
    if type(amplitude) == pd.core.series.Series:
        amplitude = amplitude[events_df.index].tolist()
        
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
                    amplitudes.append([amplitude]*len(onsets[-1]))
                elif type(amplitude) == str:
                    amplitudes.append(c_df.loc[:,amplitude].tolist())
                if type(duration) in (int,float):
                    durations.append([duration]*len(onsets[-1]))
                elif type(duration) == str:
                    durations.append(c_df.loc[:,duration].tolist())
    elif type(condition_spec) == str:
        group_df = events_df
        conditions.append(condition_spec)
        onsets.append(group_df.loc[:,onset_column].tolist())
        if type(amplitude) in (int,float):
            amplitudes.append([amplitude]*len(onsets[-1]))
        elif type(amplitude) == str:
            amplitudes.append(group_df.loc[:,amplitude].tolist())
        elif type(amplitude) == list:
            amplitudes.append(amplitude)
        if type(duration) in (int,float):
            durations.append([duration]*len(onsets[-1]))
        elif type(duration) == str:
            durations.append(group_df.loc[:,duration].tolist())
        elif type(duration) == list:
            durations.append(duration)
    # ensure that each column added is all numeric
    for attr in [durations, amplitudes, onsets]:
        assert np.issubdtype(np.array(attr[-1]).dtype, np.number) 
        assert pd.isnull(attr[-1]).sum() == 0

# specific task functions
def get_ANT_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    events_df.trial_type = [c+'_'+f for c,f in 
                            zip(events_df.cue, events_df.flanker_type)]
    
    ####################################################################################
    #################### MODIFIED IN NEWEST MODELING ###################################
    ####################################################################################
    #generate parametric regressors for 
    events_df['cue_parametric'] = -1
    events_df.loc[events_df.cue=='double', 'cue_parametric'] = 1

    events_df['congruency_parametric'] = -1
    events_df.loc[events_df.flanker_type=='incongruent', 'congruency_parametric'] = 1

    get_ev_vars(output_dict, events_df, 
            condition_spec='cue_parametric', 
            duration='duration', 
            amplitude='cue_parametric',
            subset='junk==False')

    get_ev_vars(output_dict, events_df, 
            condition_spec='congruency_parametric', 
            duration='duration', 
            amplitude='congruency_parametric',
            subset='junk==False')
    
    #Task>baseline regressor    
    get_ev_vars(output_dict, events_df, 
            condition_spec='task',
            col='trial_type',
            duration='duration', 
            subset='junk==False')
    
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')

    #RT regressors
    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                condition_spec='response_time', 
                duration='group_RT', 
                amplitude='response_time',
                subset='junk==False')
    else:
        normalize_rt(events_df, 'trial_type')
        get_ev_vars(output_dict, events_df, 
                condition_spec=[('spatial_congruent', 'spatial_congruent_RT'),
                                ('spatial_incongruent', 'spatial_incongruent_RT'),
                                ('double_congruent', 'double_congruent_RT'),
                                ('double_incongruent', 'double_incongruent_RT')],
                col='trial_type',
                amplitude='response_time',
                duration='group_RT',
                subset='junk==False')
        
#     # cue type - OLD METHOD, REDUNDANT w/ TRIAL+PARAMETRIC REGRESSORS
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec=[('spatial_congruent', 'spatial_congruent'),
#                                 ('spatial_incongruent', 'spatial_incongruent'),
#                                 ('double_congruent', 'double_congruent'),
#                                 ('double_incongruent', 'double_incongruent')],
#                 col='trial_type',
#                 duration='duration',
#                 subset='junk==False')
    
    return output_dict


def get_CCTHot_EVs(events_df, regress_rt):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
#     #task
#     get_ev_vars(output_dict, events_df,
#                 condition_spec='OLD_task',
#                 duration='block_duration',
#                 subset='junk==False and trial_id=="stim"') 
#     #add main parametric regressors: EV and risk
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='OLD_EV', 
#                 duration='block_duration', 
#                 amplitude='EV',
#                 subset='junk==False and trial_id=="stim"')
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='OLD_risk',
#                 duration='block_duration', 
#                 amplitude='risk',
#                 subset='junk==False and trial_id=="stim"')
#     #other regressors
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='OLD_num_click_in_round', 
#                 duration='block_duration', 
#                 amplitude='num_click_in_round',
#                 subset='junk==False and trial_id=="stim"')

#     # set the onset of the feedback at the time of the response - not the time of the trial beginning
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec=[(1,'OLD_reward'), (0,'OLD_punishment')], 
#                 duration=0,
#                 onset_column='movement_onset',
#                 col='feedback',
#                 subset='junk==False and trial_id=="stim"')

    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='group_RT', ############# changed in newest modeling
                    amplitude='response_time',
                    subset='junk==False')
        



    ####################################################################################
    #################### ADDED IN NEWEST MODELING ######################################
    ####################################################################################
    #build up trial regressor
    counter = 0
    round_grouping = []
    end_round_idx = events_df.index[events_df.total_cards.notnull()]
    for i in range(len(events_df.trial_id)):
        if i in end_round_idx:
            round_grouping.append(counter)
            counter+=1
        else:
            round_grouping.append(counter)
    events_df.insert(0, "round_grouping", round_grouping, True)

    round_start_idx = [0] + [x+1 for x in end_round_idx]

    task_on = [False for i in range(len(events_df.trial_id))]
    for idx in round_start_idx[:-1]:
        task_on[idx] = True

    events_df.insert(0, "round_on", task_on, True)

    round_dur = np.zeros(len(events_df.trial_id))
    for group_num in range(len(events_df.round_grouping.unique())):
        for idx in events_df.index[events_df.round_grouping==group_num].values:
            round_dur[idx] = np.sum(events_df.duration[events_df.round_grouping==np.float(group_num)])
    events_df.insert(0, "round_duration", round_dur, True)

    #add full trial length regressor
    get_ev_vars(output_dict, events_df, 
            condition_spec='task', 
            duration='round_duration',
            amplitude=1,
            subset="junk==False and round_on==True")

    #build up loss regressor
    loss_idx = events_df.index[(events_df.total_cards.notnull()) & (events_df.action=='draw_card')]
    lost_bool = [False for i in range(len(events_df.trial_id))]
    for idx in loss_idx:
        lost_bool[idx] = True
    events_df.insert(0, "lost_bool", lost_bool, True)

    #add loss event regressor
    get_ev_vars(output_dict, events_df, 
        condition_spec='loss_event',
        duration=1,
        amplitude=1,
        subset="junk==False and lost_bool==True")

    #button press regressor
    events_df['button_onset'] = events_df.onset+events_df.duration

    get_ev_vars(output_dict, events_df, 
        condition_spec='button_press',
        onset_column='button_onset',
        duration=1,
        amplitude=1,
        subset="junk==False")

    #positive_draw regressor
    get_ev_vars(output_dict, events_df, 
        condition_spec='positive_draw',
        onset_column='button_onset',
        duration=1,
        amplitude='EV',
        subset="junk==False and action=='draw_card' and feedback==1")

    #create trial-long gain and feedback regressors
    #gain
    mean_gain = np.mean(events_df[events_df.round_on==True].gain_amount.values) #demeaned based on gain for each trial, not each card
    demeaned_gain = [i - mean_gain for i in events_df.gain_amount.values]
    events_df.insert(0, "demeaned_gain", demeaned_gain, True)

    get_ev_vars(output_dict, events_df, 
        condition_spec='trial_gain',
    #     onset_column='onset',
        duration="round_duration",
        amplitude='demeaned_gain',
        subset="junk==False and round_on==True")

    #loss
    mean_loss = np.mean(events_df[events_df.round_on==True].loss_amount.values) #demeaned based on loss for each trial, not each card
    demeaned_loss = [i - mean_loss for i in events_df.loss_amount.values]
    events_df.insert(0, "demeaned_loss", demeaned_loss, True)

    get_ev_vars(output_dict, events_df, 
        condition_spec='trial_loss',
    #     onset_column='onset',
        duration="round_duration",
        amplitude='demeaned_loss',
        subset="junk==False and round_on==True")


    #build up loss value and feedback regressors
    roundsums=[]
    for group_num in events_df.round_grouping.unique():
        chunk = events_df[events_df.round_grouping==group_num].reset_index()
        roundsum = 0
        for i in range(len(chunk.feedback)):
            if chunk.feedback[i]==1:
                roundsum += chunk.gain_amount[i]
            elif chunk.feedback[i]==0:
                roundsum += chunk.loss_amount[i]
        roundsums.append(roundsum)

    feedback_values = events_df.gain_amount.to_numpy().copy()
    loss_indices = events_df.index[events_df.feedback==0] #get losses
    for loss_idx in loss_indices:
        round_idx = events_df.round_grouping[loss_idx]
        feedback_values[loss_idx] = roundsums[round_idx]

    events_df.insert(0, "feedback_values", feedback_values, True)
    events_df.insert(0, "demeaned_feedback_values", feedback_values-np.mean(feedback_values), True)

    #create feedback regressor - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#         condition_spec='feedback',
#         onset_column='button_onset',
#         duration=1,
#         amplitude='demeaned_feedback_values',
#         subset="junk==False and action=='draw_card'")
    
    demeaned_loss_array = np.zeros(len(events_df.feedback_values))
    demeaned_losses = events_df.feedback_values[loss_indices].copy() - np.mean(events_df.feedback_values[loss_indices])
    for idx in loss_indices:
        demeaned_loss_array[idx] = demeaned_losses[idx]
    events_df.insert(0, "demeaned_trial-cumulative_loss_values", demeaned_loss_array, True)
    
    #create loss regressor
    get_ev_vars(output_dict, events_df, 
        condition_spec='loss',
        onset_column='button_onset',
        duration=1, 
        amplitude='demeaned_trial-cumulative_loss_values',
        subset="junk==False and action=='draw_card' and feedback==0")
    
    return output_dict

def get_discountFix_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # regressors of interest
    """
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('larger_later', 'larger_later'),
                                ('smaller_sooner', 'smaller_sooner')],
                col='trial_type',
                duration='duration',
                subset='junk == False')
    """
#     SUBJECTIVE CHOICE VALUE - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='subjective_choice_value', 
#                 duration='duration', 
#                 amplitude='subjective_choice_value',
#                 subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')],
                col='junk', 
                duration='duration')
    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='group_RT', 
                    amplitude='response_time',
                    subset='junk==False')
        
    ####################################################################################
    #################### ADDED IN NEWEST MODELING ######################################
    ####################################################################################
    #trial regressor for task > baseline
    get_ev_vars(output_dict, events_df, 
                condition_spec='task', 
                duration='duration', 
                amplitude=1,
                subset='junk==False')
    #choice regressor
    choices = [-1] * len(events_df.trial_type)
    larger_indices = events_df.index[events_df.trial_type=='larger_later'].tolist()
    for idx in larger_indices:
        choices[idx] = 1
    events_df.insert(0, 'choice_contrast', choices, True)
    
    get_ev_vars(output_dict, events_df, 
                condition_spec='choice', 
                duration='duration', 
                amplitude='choice_contrast',
                subset='junk==False')   
    
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
                duration='duration',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration') 
    ####################################################################################
    #################### MODIFIED IN NEWEST MODELING ######################################
    ####################################################################################
    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                condition_spec='response_time', 
                duration='group_RT', 
                amplitude='response_time',
                subset='junk==False')
    else:
        normalize_rt(events_df, 'condition')
        get_ev_vars(output_dict, events_df, 
                condition_spec=[('AX', 'AX_RT'),
                                ('AY', 'AY_RT'),
                                ('BX', 'BX_RT'),
                                ('BY', 'BY_RT')],
                col='condition',
                amplitude='response_time',
                duration='group_RT',
                subset='junk==False')
        

    #trial regressor for task > baseline - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='task', 
#                 duration='duration', 
#                 amplitude=1,
#                 subset='junk==False')
    
    return output_dict

def get_manipulation_EVs(events_df, regress_rt=True): 
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
      
    get_ev_vars(output_dict, events_df,
               condition_spec = [('cue', 'task')],
               col = 'trial_id',
               duration = 10,
               subset='trial_type!="no_stim" and junk==False')
    
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
           
    #replace strings with parametric regressors adusted for frequency of trial type 
    cue_count = events_df['which_cue'].value_counts()    
    events_df.which_cue = events_df.which_cue.replace('LATER', (cue_count['NOW'])/(cue_count['LATER'])) 
    events_df.which_cue = events_df.which_cue.replace('NOW', -1) 
    
    # cue regressor 
    get_ev_vars(output_dict, events_df, 
               condition_spec = [('cue', 'cue')],
               col = 'trial_id',
               duration = 'duration',
               amplitude = 'which_cue',
               subset='trial_type!="no_stim" and junk==False')
    
    #demean response rating
    response_mean = np.nanmean(events_df['response'])
    response_mean
    events_df['demeaned_response'] = events_df.response - response_mean
    
    get_ev_vars(output_dict, events_df, 
           condition_spec = 'rating',
           duration = 'duration',
           amplitude = 'demeaned_response',
           subset='trial_type!="no_stim" and junk==False and trial_id=="current_rating"')
    
    #demean RT
    events_df["demeaned_rt"] =  events_df.response_time - np.nanmean(events_df.response_time)
    
    get_ev_vars(output_dict, events_df, 
       condition_spec = 'response_time',
       duration = 'group_RT',
       amplitude = 'demeaned_rt',
       subset='trial_type!="no_stim" and junk==False and trial_id=="current_rating"')
               
    
    #demean probe regressor 
    probe_ratio = events_df['stim_type'].value_counts()
    events_df.stim_type = events_df.stim_type.replace('neutral', -(probe_ratio['valence'])/(probe_ratio['neutral']))
    events_df.stim_type = events_df.stim_type.replace('valence', 1)
    
    get_ev_vars(output_dict, events_df, 
                condition_spec = [('probe', 'probe')],
                col = 'trial_id',
               duration = 'duration',
               amplitude = 'stim_type', 
               subset='trial_type!="no_stim" and junk==False')
        
    return output_dict
   

def get_motorSelectiveStop_EVs(events_df, regress_rt=True):
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
    ####################################################################################
    #################### MODIFIED IN NEWEST MODELING ###################################
    ####################################################################################
    
    # create 1 RT regressor for all go trials, 1 for stop failures
    events_df['simplified_trial_type'] = 'go'
    events_df.loc[events_df.trial_type=='crit_stop_failure', 'simplified_trial_type'] = 'crit_stop_failure'

    normalize_rt(events_df, 'simplified_trial_type')
    get_ev_vars(output_dict, events_df, 
            condition_spec=[('go','go_RT'), 
                        ('crit_stop_failure', 'crit_stop_failure_RT')],
            col='simplified_trial_type',
            amplitude='response_time',
            duration='group_RT',
            subset='junk==False')

#### OLD RT REGRESSORS
#     if regress_rt == True:
#         normalize_rt(events_df)
#         get_ev_vars(output_dict, events_df, 
#                 condition_spec='response_time', 
#                 duration='group_RT', 
#                 amplitude='response_time',
#                 subset='junk==False and trial_type!="crit_stop_success"')
#     else:
#         normalize_rt(events_df, 'trial_type')
#         get_ev_vars(output_dict, events_df, 
#                 condition_spec=[('crit_go','crit_go_RT'), 
#                             ('crit_stop_failure', 'crit_stop_failure_RT'),
#                             ('noncrit_signal', 'noncrit_signal_RT'),
#                             ('noncrit_nosignal', 'noncrit_nosignal_RT')],
#                 col='trial_type',
#                 amplitude='response_time',
#                 duration='group_RT',
#                 subset='junk==False')
        

#     #trial regressor for task > baseline - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='task', 
#                 duration='duration', 
#                 amplitude=1,
#                 subset='junk==False')
    
    return output_dict
    
def get_stopSignal_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
     # task regressor
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('go','go'), 
                            ('stop_success', 'stop_success'), 
                            ('stop_failure', 'stop_failure')],
                col='trial_type', 
                duration='duration',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
    ####################################################################################
    #################### MODIFIED IN NEWEST MODELING ###################################
    ####################################################################################
#     if regress_rt == True: ######MODIFED - stopSignal rt are normalized by trial_type
#         normalize_rt(events_df)
#         get_ev_vars(output_dict, events_df, 
#                 condition_spec='response_time', 
#                 duration='group_RT', 
#                 amplitude='response_time',
#                 subset='junk==False and trial_type!="stop_success"')
#     else:
    normalize_rt(events_df, 'trial_type') ########  - stopSignal rt are normalized by trial_type
    get_ev_vars(output_dict, events_df, 
            condition_spec=[('go','go_RT'), 
                        ('stop_failure', 'stop_failure_RT')],
            col='trial_type',
            amplitude='response_time',
            duration='group_RT',
            subset='junk==False')


#     #trial regressor for task > baseline - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='task', 
#                 duration='duration', 
#                 amplitude=1,
#                 subset='junk==False')
    return output_dict

def get_stroop_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # task regressor
    # contrast regressor - OLD
#     get_ev_vars(output_dict, events_df,
#                 condition_spec=[('incongruent', 'incongruent'),
#                                ('congruent', 'congruent')],
#                 col='condition',
#                 duration='duration',
#                 subset='junk==False')

    ####################################################################################
    #################### MODIFIED IN NEWEST MODELING ###################################
    ####################################################################################
    #parametric congruency regressor
    events_df['congruency_parametric'] = -1
    events_df.loc[events_df.trial_type=='incongruent', 'congruency_parametric'] = 1
    
    get_ev_vars(output_dict, events_df, 
        condition_spec='congruency_parametric', 
        duration='duration', 
        amplitude='congruency_parametric',
        subset='junk==False')
    
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration') 
    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                condition_spec='response_time', 
                duration='group_RT', 
                amplitude='response_time',
                subset='junk==False')
    else:
        normalize_rt(events_df, 'condition')
        get_ev_vars(output_dict, events_df, 
                condition_spec=[('incongruent', 'incongruent_RT'),
                               ('congruent', 'congruent_RT')],
                col='condition',
                amplitude='response_time',
                duration='group_RT',
                subset='junk==False')
        

    #trial regressor for task > baseline
    get_ev_vars(output_dict, events_df, 
                condition_spec='task', 
                duration='duration', 
                amplitude=1,
                subset='junk==False')
    
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
                    duration='group_RT', 
                    amplitude='response_time',
                    subset='junk==False')
        
    ####################################################################################
    #################### ADDED IN NEWEST MODELING ######################################
    ####################################################################################
    #trial regressor for task > baseline
    get_ev_vars(output_dict, events_df, 
                condition_spec='task', 
                duration='duration', 
                amplitude=1,
                subset='junk==False')
    
    return output_dict

    
def get_twoByTwo_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    events_df.trial_type = ['cue_'+c if c is not np.nan else 'task_'+t \
                            for c,t in zip(events_df.cue_switch, events_df.task_switch)]
    events_df.trial_type.replace('cue_switch', 'task_stay_cue_switch', inplace=True)
    
    # trial type contrasts
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('task_switch', 'task_switch_900'),
                                ('task_stay_cue_switch', 'task_stay_cue_switch_900'),
                               ('cue_stay', 'cue_stay_900')],
                col='trial_type',
                duration='duration',
                subset="CTI==900 and junk==False")
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('task_switch', 'task_switch_100'),
                                ('task_stay_cue_switch', 'task_stay_cue_switch_100'),
                               ('cue_stay', 'cue_stay_100')],
                col='trial_type',
                duration='duration',
                subset="CTI==100 and junk==False")
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
    ####################################################################################
    #################### MODIFIED IN NEWEST MODELING ###################################
    ####################################################################################
    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='group_RT', 
                    amplitude='response_time',
                    subset='junk==False')
    else:
        # normalize RT separately for each condition and CTI
        subset_100 = events_df.query('CTI == 100')
        subset_900 = events_df.query('CTI == 900')
        normalize_rt(subset_100, 'trial_type')
        normalize_rt(subset_900, 'trial_type')
        events_df.loc[subset_100.index, 'response_time'] = subset_100.response_time
        events_df.loc[subset_900.index, 'response_time'] = subset_900.response_time
        get_ev_vars(output_dict, events_df, 
                    condition_spec=[('task_switch', 'task_switch_100_RT'),
                                    ('task_stay_cue_switch', 'task_stay_cue_switch_100_RT'),
                                   ('cue_stay', 'cue_stay_100_RT')],
                    col='trial_type',
                    duration='group_RT', 
                    amplitude='response_time',
                    subset='CTI==100 and junk==False')
        get_ev_vars(output_dict, events_df, 
                    condition_spec=[('task_switch', 'task_switch_900_RT'),
                                    ('task_stay_cue_switch', 'task_stay_cue_switch_900_RT'),
                                   ('cue_stay', 'cue_stay_900_RT')],
                    col='trial_type',
                    duration='group_RT', 
                    amplitude='response_time',
                    subset='CTI==900 and junk==False')


#     #trial regressor for task > baseline - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='task', 
#                 duration='duration', 
#                 amplitude=1,
#                 subset='junk==False')   

    return output_dict

def get_WATT3_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }

    ####################################################################################
    ######################### Modified IN NEWEST MODELING ##############################
    ####################################################################################
    
    events_df.block_duration = events_df.block_duration/1000
    
    # planning conditions
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec=[('PA_with_intermediate','plan_PA_with'),
#                                 ('PA_without_intermediate','plan_PA_without')],
#                 col='condition', 
#                 duration='duration', 
#                 subset="planning==1")

    #planning event
    get_ev_vars(output_dict, events_df, 
                condition_spec='planning_event',
                duration='block_duration', 
                subset="planning==1")
    #parametric planning event
    events_df.condition = events_df.condition.replace('PA_without_intermediate', -1)
    events_df.condition = events_df.condition.replace('PA_with_intermediate', 1)
    get_ev_vars(output_dict, events_df, 
            condition_spec='planning_parametric',
            duration='block_duration', 
            amplitude='condition',
            subset="planning==1")

    # nuisance regressors
    #movement
    get_ev_vars(output_dict, events_df, 
                condition_spec='movement', 
                duration=1,
                onset_column='movement_onset')
    
    #parametric movement ########## Look into this
    get_ev_vars(output_dict, events_df, 
                condition_spec='movement_parametric', 
                duration=1,
                amplitude='condition',
                onset_column='movement_onset')
    
    #feedback
    get_ev_vars(output_dict, events_df, 
                condition_spec='feedback', 
                duration='block_duration',
                subset="trial_id=='feedback'")
    #parametric feedback
    get_ev_vars(output_dict, events_df, 
                condition_spec='feedback_parametric', 
                duration='block_duration',
                amplitude='condition',
                subset="trial_id=='feedback'")
    
    if regress_rt == True:
        normalize_rt(events_df)
        get_ev_vars(output_dict, events_df, 
                condition_spec='response_time', 
                duration='group_RT', ##changed in newest modeling
                amplitude='response_time',
                subset="trial_id != 'feedback'")
    else:
        normalize_rt(events_df, 'planning')
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='group_RT', ##changed in newest modeling
                    amplitude='response_time',
                    subset="trial_id != 'feedback'")
        

    #trial regressor for task > baseline
    #build up trial regressor
    counter = 0
    round_grouping = []
    end_round_idx = events_df.index[events_df.trial_id=='feedback']
    for i in range(len(events_df.trial_id)):
        if i in end_round_idx:
            round_grouping.append(counter)
            counter+=1
        else:
            round_grouping.append(counter)
    events_df.insert(0, "round_grouping", round_grouping, True)
    
    round_start_idx = [0] + [x+1 for x in end_round_idx]
    
    round_dur = np.zeros(len(events_df.trial_id))
    for group_num in range(len(events_df.round_grouping.unique())):
        for idx in events_df.index[events_df.round_grouping==group_num].values:
            round_dur[idx] = np.sum(events_df.block_duration[events_df.round_grouping==np.float(group_num)][:-1])
    events_df.insert(0, "round_duration", round_dur, True)
    
#     #add full trial length regressor - FOUND TO BE REDUNDANT
#     get_ev_vars(output_dict, events_df, 
#             condition_spec='task', 
#             duration='round_duration',
#             amplitude=1,
#             subset="junk==False and planning==1") 
    
    return output_dict

# def get_base_EVs(events_df):
#     output_dict = {
#         'conditions': [],
#         'onsets': [],
#         'durations': [],
#         'amplitudes': []
#         }
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec='trial',
#                 duration='duration')
#     get_ev_vars(output_dict, events_df, 
#                 condition_spec=[(True, 'junk')], 
#                 col='junk', 
#                 duration='duration')   
#     return output_dict

def get_beta_series(events_df, regress_rt=True):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    for i, row in events_df.iterrows():
        if row.junk == False:
            output_dict['conditions'].append('trial_%s' % str(i+1).zfill(3))
            output_dict['onsets'].append([row.onset])
            output_dict['durations'].append([row.duration])
            output_dict['amplitudes'].append([1])
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict
    
# How to model RT
# For each condition model responses with constant duration 
# (average RT across subjects or block duration)
# RT as a separate regressor for each onset, constant duration, 
# amplitude as parameteric regressor (function of RT)
def parse_EVs(events_df, task, regress_rt=True):
    if task == "ANT":
        EV_dict = get_ANT_EVs(events_df, regress_rt)
    elif task == "CCTHot": 
        EV_dict = get_CCTHot_EVs(events_df, regress_rt)
    elif task == "discountFix": 
        EV_dict = get_discountFix_EVs(events_df, regress_rt)
    elif task == "DPX":
        EV_dict = get_DPX_EVs(events_df, regress_rt)
    elif task == 'manipulationTask': 
        EV_dict = get_manipulation_EVs(events_df, regress_rt)
    elif task == "motorSelectiveStop": 
        EV_dict = get_motorSelectiveStop_EVs(events_df)
    elif task == 'surveyMedley':
        EV_dict = get_surveyMedley_EVs(events_df, regress_rt)
    elif task == "stopSignal":
        EV_dict = get_stopSignal_EVs(events_df)
    elif task == "stroop":
        EV_dict = get_stroop_EVs(events_df, regress_rt)
    elif task == "twoByTwo":
        EV_dict = get_twoByTwo_EVs(events_df, regress_rt)
    elif task == "WATT3":
        EV_dict = get_WATT3_EVs(events_df, regress_rt)
    # covers generic conversion of events_df into trial design file
    elif task == 'beta':
        EV_dict = get_beta_series(events_df, regress_rt)
    return EV_dict

    
