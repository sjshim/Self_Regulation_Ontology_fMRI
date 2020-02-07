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
    events_dir = join(file_dir, '../behavioral_data/aim1/event_files')
    subj = subj.replace('sub-','')
    # get event file
    ev_file = glob(join(events_dir,'*%s*%s*' % (subj, task)))[0]
    task_fmri_files = glob(join(fmri_dir, '*%s*' % subj,'*', # add * to account for ses-dir
                                'func','*%s*bold*' % task))  
    task_fmri_dir = dirname(task_fmri_files[0])
    base_name = basename(task_fmri_files[0]).split('_bold')[0]
    new_events_file = join(task_fmri_dir, base_name+'_events.tsv')
    shutil.copyfile(ev_file, new_events_file)
    return new_events_file

def move_EV2(subj, task, fmri_dir):
    file_dir = path.dirname(__file__)
    events_dir = join(file_dir, '../behavioral_data/aim2/event_files')
    subj = subj.replace('sub-','')
    # get event file
    ev_file = glob(join(events_dir,'%s_%s*' % (subj, task)))[0]
    task_fmri_files = glob(join(fmri_dir, 'sub-%s' % subj, 
                                'func','*%s*bold*' % task))
    task_fmri_dir = dirname(task_fmri_files[0])
    base_name = basename(task_fmri_files[0]).split('_bold')[0]
    new_events_file = join(task_fmri_dir, base_name+'_events.tsv')
    shutil.copyfile(ev_file, new_events_file)
    return new_events_file
    
def move_EVs(fmri_dir, tasks, overwrite=True, verbose=False):
    created_files = []
    total_transfers = {t:0 for t in tasks}
    for subj_file in sorted(glob(join(fmri_dir,'sub-*'))):
        subj = basename(subj_file)
        if verbose: print('Transferring subject %s' % subj)
        for task in tasks:
            bold_files = glob(join(subj_file, 'func', '*task-%s*bold.nii.gz' % task)) # add * to account for ses-dir 
            assert len(bold_files) <= 1, "%s bold files found for %s_%s" % (len(bold_files), subj, task)
            if len(bold_files) == 1:
                event_files = glob(join(subj_file, 'func', '*%s*events.tsv' % task))
                if overwrite==True or len(event_files)==0:
                    try:
                        name = move_EV(subj, task, fmri_dir)
                        created_files.append(name)
                        total_transfers[task] += 1
                    except IndexError:
                        print('Move_EV failed for the %s: %s' % (subj, task))
            else:
                print('**** No %s bold found for %s' % (task, subj))
    if verbose:
        print('\n'.join(created_files))
        print(total_transfers)

def move_EVs2(fmri_dir, tasks, overwrite=True, verbose=False):
    created_files = []
    total_transfers = {t:0 for t in tasks}
    for subj_file in sorted(glob(join(fmri_dir,'sub-*'))):
        subj = basename(subj_file)
        if verbose: print('Transferring subject %s' % subj)
        for task in tasks:
            bold_files = glob(join(subj_file, 'func', '*task-%s*bold.nii.gz' % task)) # add * to account for ses-dir 
            assert len(bold_files) <= 1, "%s bold files found for %s_%s" % (len(bold_files), subj, task)
            if len(bold_files) == 1:
                event_files = glob(join(subj_file, 'func', '*%s*events.tsv' % task))
                if overwrite==True or len(event_files)==0:
                    try:
                        name = move_EV2(subj, task, fmri_dir)
                        created_files.append(name)
                        total_transfers[task] += 1
                    except IndexError:
                        print('Move_EV failed for the %s: %s' % (subj, task))
            else:
                print('**** No %s bold found for %s' % (task, subj))
    if verbose:
        print('\n'.join(created_files))
        print(total_transfers)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
    parser.add_argument('--data_dirs', nargs='+', default=['/aim1/data', '/aim2/data'], help='BIDS directory')
    parser.add_argument('--tasks', default=None, nargs="+")
    parser.add_argument('--overwrite_event', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--aim', nargs='+', default=['aim1', 'aim2'])
    args = parser.parse_args()
    
    overwrite_event = args.overwrite_event
    verbose = args.verbose
    data_dirs = args.data_dirs
    aims=args.aim

    for aim in aims:
        if aim=='aim1':        
            if args.tasks:
                task_list = args.tasks
            else:
                task_list = ['ANT', 'CCTHot', 'discountFix', 'DPX',
                  'motorSelectiveStop', 'stopSignal', 
                  'stroop', 'twoByTwo', 'WATT3']
            move_EVs(data_dirs[0], task_list, overwrite_event, verbose=verbose)
        elif aim=='aim2':
            if len(data_dirs)>1:
                data_dir = data_dirs[1]
            else:
                data_dir = data_dirs[0]
            if args.tasks:
                task_list = args.tasks
            else:
                task_list = [ 'discountFix',
                            'motorSelectiveStop',
                            'stopSignal', 'manipulationTask' ] 
            print(data_dir)
            move_EVs2(data_dir, task_list, overwrite_event, verbose=verbose)

