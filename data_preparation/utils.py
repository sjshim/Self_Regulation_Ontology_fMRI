from expanalysis.experiments.jspsych_processing import ANT_HDDM, EZ_diffusion, \
    get_post_error_slow, group_decorate
import pandas as pd

# function to correct processing of a few problematic files
# need to change time_elapsed to reflect the fact that fmri triggers were
# sent outto quickly (at 8 times the rate), thus starting the scan 14 TRs
# early. Those 14 TRs of data therefore need to be thrown out, which is
# accomplished by setting the "0" of the scan 14 TRs later
def get_timing_correction(filey, TR=680, n_TRs=14):
    problematic_files = ['s568_MotorStop.csv', 's568_Stroop.csv', 
                         's568_SurveyMedley.csv', 's568_DPX.csv',
                         's568_Discount.csv',
                         's556_MotorStop.csv', 's556_Stroop.csv', 
                         's556_SurveyMedley.csv', 's556_DPX.csv',
                         's556_Discount.csv',
                         's561_WATT.csv', 's561_ANT.csv', 
                         's561_TwoByTwo.csv', 's561_CCT.csv',
                         's561_StopSignal.csv',]
    tr_correction = TR * n_TRs
    if filey in problematic_files:
        return tr_correction
    else:
        return 0
    


# DV functions for fmri tasks if not already created in expfactory-analysis
@group_decorate(group_fun = ANT_HDDM)
def calc_ANT_DV(df, dvs = {}):
    """ Calculate dv for attention network task: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'flanker_shift', df.flanker_type.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True')
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'flanker_type'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # Get three network effects
    cue_rt = df_correct.groupby('cue').rt.median()
    flanker_rt = df_correct.groupby('flanker_type').rt.median()
    cue_acc = df.groupby('cue').correct.mean()
    flanker_acc = df.groupby('flanker_type').correct.mean()
    
    dvs['orienting_rt'] = {'value':  (cue_rt.loc['double'] - cue_rt.loc['spatial']), 'valence': 'Pos'}
    dvs['conflict_rt'] = {'value':  (flanker_rt.loc['incongruent'] - flanker_rt.loc['congruent']), 'valence': 'Neg'}
    dvs['orienting_acc'] = {'value':  (cue_acc.loc['double'] - cue_acc.loc['spatial']), 'valence': 'NA'}
    dvs['conflict_acc'] = {'value':  (flanker_acc.loc['incongruent'] - flanker_acc.loc['congruent']), 'valence': 'Pos'}
    
    # DDM equivalents
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    if set(['EZ_drift_congruent', 'EZ_drift_incongruent']) <= set(dvs.keys()):
        cue_ddm = EZ_diffusion(df,'cue')
        for param in ['drift','thresh','non_decision']:
            dvs['orienting_EZ_' + param] = {'value':  cue_ddm['EZ_' + param + '_double']['value'] - cue_ddm['EZ_' + param + '_spatial']['value'], 'valence': param_valence[param]}
            dvs['conflict_EZ_' + param] = {'value':  dvs['EZ_' + param + '_incongruent']['value'] - dvs['EZ_' + param + '_congruent']['value'], 'valence': param_valence[param]}

    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_congruent', 'hddm_' + param + '_incongruent']) <= set(dvs.keys()):
            dvs['conflict_hddm_' + param] = {'value':  dvs['hddm_' + param + '_incongruent']['value'] - dvs['hddm_' + param + '_congruent']['value'], 'valence': param_valence[param]}
        if set(['hddm_' + param + '_double', 'hddm_' + param + '_spatial']) <= set(dvs.keys()):
            dvs['orienting_hddm_' + param] = {'value':  dvs['hddm_' + param + '_double']['value'] - dvs['hddm_' + param + '_spatial']['value'], 'valence': param_valence[param]}
    # remove unnecessary cue dvs
    for key in list(dvs.keys()):
        if any(x in key for x in ['double', 'spatial']):
            del dvs[key]
            
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).correct.mean()
    
    try:
        seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
            (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
        seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
            (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
        dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'}
        dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'}
    except KeyError:
        pass
    
    description = """
    DVs for "alerting", "orienting" and "conflict" attention networks are of primary
    interest for the ANT task, all concerning differences in RT. 
    Alerting is defined as nocue - double cue trials. Positive values
    indicate the benefit of an alerting double cue. Orienting is defined as center - spatial cue trials.
    Positive values indicate the benefit of a spatial cue. Conflict is defined as
    incongruent - congruent flanker trials. Positive values indicate the benefit of
    congruent trials (or the cost of incongruent trials). RT measured in ms and median
    RT are used for all comparisons.
    
    DDM comparisons are used for conflict conditions. Drift is expected to be lower for the more difficult
    incongruent condition (incongruent-congruent is negative). Thus higher values are good, meaning there
    is less of a disparity. Thresholds may be higher in conflict conditions. Higher values (incongruent-congruent)
    are also good, as they indicate adapatible changes in the amount of evidence needed based on the 
    difficulty of the trial.
    """
    return dvs, description