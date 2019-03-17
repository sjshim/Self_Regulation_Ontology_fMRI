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

def get_median_rts(task_dfs):
    task_50th_rts = {task: df.rt[df.rt>0].quantile(.5) for task,df in task_dfs.items()}
    # special cases handled below
    # ** twoByTwo **
    median_cue_length = task_dfs['twobytwo'].CTI.quantile(.5)
    task_50th_rts['twobytwo'] += median_cue_length
    
    # ** WATT3 **
    WATT_df = task_dfs['ward_and_allport'].query('exp_stage == "test"')
    # get the first move times (plan times)
    plan_times = WATT_df.query('trial_id == "to_hand" and num_moves_made==1').rt
    # get other move times
    move_times = WATT_df.query('not (trial_id == "to_hand" and num_moves_made==1)')
    # drop feedback
    move_times = move_times.query('trial_id != "feedback"').rt
    task_50th_rts['ward_and_allport'] = {'planning_time': plan_times.quantile(.5),
                                         'move_time': move_times.quantile(.5)}
    return task_50th_rts

def get_survey_items_order():
    
    """Function which returns dictionary with ordering id (Q01-Q40) assigned to each question. 
    This dictionary can be further used to map all quesion to their unique (template) order, therefore, to obtain the same order of beta vales for each person
    Author: Karolina Finc
    """
    
    grit_items = [
        'New ideas and projects sometimes distract me from previous ones.',
        'Setbacks don\'t discourage me.',
        'I have been obsessed with a certain idea or project for a short time but later lost interest.',
        'I am a hard worker.',
        'I often set a goal but later choose to pursue a different one.',
        'I have difficulty maintaining my focus on projects that take more than a few months to complete.',
        'I finish whatever I begin.',
        'I am diligent.'
    ]

    brief_items = [
        'I am good at resisting temptation.',
        'I have a hard time breaking bad habits.',
        'I am lazy.',
        'I say inappropriate things.',
        'I do certain things that are bad for me, if they are fun.',
        'I refuse things that are bad for me.',
        'I wish I had more self-discipline.',
        'People would say that I have iron self-discipline.',
        'Pleasure and fun sometimes keep me from getting work done.',
        'I have trouble concentrating.',
        'I am able to work effectively toward long-term goals.',
        'Sometimes I can\'t stop myself from doing something, even if I know it is wrong.',
        'I often act without thinking through all the alternatives.'
     ]

    future_time_items = [
        'Many opportunities await me in the future.',
        'I expect that I will set many new goals in the future.',
        'My future is filled with possibilities.',
        'Most of my life lies ahead of me.',
        'My future seems infinite to me.',
        'I could do anything I want in the future.',
        'There is plenty of time left in my life to make new plans.',
        'I have the sense that time is running out.',
        'There are only limited possibilities in my future.',
        'As I get older, I begin to experience time as limited.'
     ]

    upps_items = [
        "Sometimes when I feel bad, I can't seem to stop what I am doing even though it is making me feel worse.",
        'Others would say I make bad choices when I am extremely happy about something.',
        'When I get really happy about something, I tend to do things that can have bad consequences.',
        'When overjoyed, I feel like I cant stop myself from going overboard.',
        'When I am really excited, I tend not to think of the consequences of my actions.',
        'I tend to act without thinking when I am really excited.'
    ]
    
    impulse_venture_items = [
        'Do you welcome new and exciting experiences and sensations even if they are a little frightening and unconventional?',
        'Do you sometimes like doing things that are a bit frightening?',
        'Would you enjoy the sensation of skiing very fast down a high mountain slope?'
    ]
    
    item_text = grit_items + brief_items + future_time_items + upps_items + impulse_venture_items
    item_id = ['Q%s' % str(i+1).zfill(2) for i in range(len(item_text))]
    item_id_map = dict(zip(item_text, item_id))

    return item_id_map