from glob import glob
from os import path
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

def get_name_map():
    name_map = {'attention_network_task': 'ANT',
            'columbia_card_task_hot': 'CCTHot',
            'discount_fixed': 'discountFix',
            'dot_pattern_expectancy': 'DPX',
            'motor_selective_stop_signal': 'motorSelectiveStop',
            'stop_signal': 'stopSignal',
            'stroop': 'stroop',
            'survey_medley': 'surveyMedley',
            'twobytwo': 'twoByTwo',
            'ward_and_allport': 'WATT3'}
    return name_map


def get_event_files(subj):
    file_dir = path.dirname(__file__)
    event_files = {}
    for subj_file in glob(path.join(file_dir, '../behavioral_data/event_files/*%s*' % subj)):
        df = pd.read_csv(subj_file, sep='\t')
        exp_id = path.basename(subj_file).split('_')[1]
        event_files[exp_id] = df
    return event_files
        
def get_processed_files(subj):
    file_dir = path.dirname(__file__)
    processed_files = {}
    for subj_file in glob(path.join(file_dir, '../behavioral_data/processed/*%s*' % subj)):
        df = pd.read_csv(subj_file)
        exp_id = path.basename(subj_file).split('_')[1]
        processed_files[exp_id] = df
    return processed_files