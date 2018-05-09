import argparse
from glob import glob
from os import path
from os.path import basename, dirname, join, exists
import shutil

# ********************************************************
# Behavioral Utility Functions
# ********************************************************

def move_EV(subj, task, fmri_dir):
    file_dir = path.dirname(__file__)
    events_dir = join(file_dir, '../behavioral_data/event_files')
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
    
def move_EVs(fmri_dir, tasks, overwrite=True, verbose=False):
    created_files = []
    for subj_file in sorted(glob(join(fmri_dir,'sub-s???'))):
        subj = basename(subj_file)
        if verbose: print('Transferring subject %s' % subj)
        for task in tasks:
            bold_files = glob(join(subj_file,'*', 'func', '*%s*bold.nii.gz' % task))
            assert len(bold_files) <= 1, "%s bold files found for %s_%s" % (len(bold_files), subj, task)
            if len(bold_files) == 1:
                event_files = glob(join(subj_file,'*', 'func', '*%s*events.tsv' % task))
                if overwrite==True or len(event_files)==0:
                    try:
                        name = move_EV(subj, task, fmri_dir)
                        created_files.append(name)
                    except IndexError:
                        print('Move_EV failed for the %s: %s' % (subj, task))
            else:
                print('**** No %s bold found' % task)
    if verbose:
        print('\n'.join(created_files))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
    parser.add_argument('-data_dir', default='/data')
    parser.add_argument('--tasks', default=None, nargs="+")
    parser.add_argument('--overwrite_event', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    overwrite_event = args.overwrite_event
    verbose = args.verbose
    data_dir = args.data_dir
    if args.tasks:
        task_list = args.tasks
    else:
      task_list = ['ANT', 'CCTHot', 'discountFix',
                   'DPX', 'motorSelectiveStop',
                   'stopSignal', 'stroop', 'surveyMedley',
                   'twoByTwo', 'WATT3']
    
    move_EVs(data_dir, task_list, overwrite_event, verbose=verbose)
