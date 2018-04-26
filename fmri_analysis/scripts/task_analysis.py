
# coding: utf-8
import argparse
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
from os.path import join
from utils.event_utils import move_EVs


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('-output_dir', default='/output', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('-data_dir',help='The directory of the preprocessed fmri'
                    'data (output of fmriprep) along with event files',
                    default='/data')
parser.add_argument('--participant_label',help='The label(s) of the participant(s)'
                   'that should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+")
parser.add_argument('--events_dir', help='The directory of the events file. If'
                    'provided, events files will be copied to data_dir before'
                    'continuing', default=None)
parser.add_argument('--tasks',help='The label(s) of the task(s)'
                   'that should be analyzed. If this parameter is not '
                   'provided all tasks should be analyzed.',
                   nargs="+")
parser.add_argument('--use_events', action='store_false', 
                    help='If included, use events file')
parser.add_argument('--ignore_rt', action='store_true', 
                    help='If included, ignore respone'
                    'time as a regressor')
parser.add_argument('--cleanup', action='store_true', 
                    help='If included, delete working directory')
parser.add_argument('--overwrite_event', action='store_true',
                    help='If included and events_dir is included, overwrite'
                    'events files')

args = parser.parse_args()
# list of subject identifiers
subject_list = args.participant_label
# list of task identifiers
if args.tasks:
    task_list = args.tasks
else:
  task_list = ['ANT', 'CCTHot', 'discountFix',
               'DPX', 'motorSelectiveStop',
               'stopSignal', 'stroop', 'surveyMedley',
               'twoByTwo', 'WATT3']

regress_rt = not args.ignore_rt
use_events = args.use_events
#### Experiment Variables
output_dir = args.output_dir
events_dir = args.events_dir
data_dir = args.data_dir
first_level_dir = '1stLevel'
working_dir = 'workingdir'
# TR of functional images
TR = .68

# ****************************************************************************
# move events files if necessary
# ***************************************************************************
# move events  
if events_dir is not None:
    move_EVs(events_dir, data_dir, task_list, args.overwrite_event)
        
# *********************************************
# ### Define helper functions
# *********************************************

# helper function to create bunch
def subjectinfo(data_dir, subject_id, task, use_events=True,
                regress_rt=True): 
    from glob import glob
    from os.path import join
    import pandas as pd
    from nipype.interfaces.base import Bunch
    from utils.event_utils import get_contrasts, parse_EVs, process_confounds
    # strip "sub" from beginning of subject_id if provided
    subject_id = subject_id.replace('sub-','')
    ## Get the Confounds File (output of fmriprep)
    # Read the TSV file and convert to pandas dataframe
    confounds_file = glob(join(data_dir,
                               'sub-%s' % subject_id,
                               '*', 'func',
                               '*%s*confounds.tsv' % task))[0]
    regressors, regressor_names = process_confounds(confounds_file)
    if use_events:
        ## Get the Events File
        # Read the TSV file and convert to pandas dataframe
        event_file = glob(join(data_dir,
                               'sub-%s' % subject_id,
                               '*', 'func',
                               '*%s*events.tsv' % task))[0]
        events_df = pd.read_csv(event_file,sep = '\t')
        # set up contrasts
        EV_dict = parse_EVs(events_df, task, regress_rt)
        subjectinfo = Bunch(subject_id=subject_id,
                            task=task,
                            conditions=EV_dict['conditions'],
                            onsets=EV_dict['onsets'],
                            durations=EV_dict['durations'],
                            amplitudes=EV_dict['amplitudes'],
                            tmod=None,
                            pmod=None,
                            regressor_names=regressor_names,
                            regressors=regressors.T.tolist())
        contrasts = get_contrasts(task, regress_rt)
        return subjectinfo, contrasts  # this output will later be returned to infosource
    else:
        subjectinfo = Bunch(subject_id=subject_id,
                            task=task,
                            tmod=None,
                            pmod=None,
                            regressor_names=regressor_names,
                            regressors=regressors.T.tolist())
        return subjectinfo, [] # this output will later be returned to infosource

def save_subjectinfo(base_directory, subject_id, task, subjectinfo, contrasts):
    from os import makedirs
    from os.path import join
    import pickle
    task_dir = join(base_directory, subject_id + '_task_' + task)
    makedirs(task_dir, exist_ok=True)
    subjectinfo_path = join(task_dir,'subjectinfo.pkl')
    pickle.dump(subjectinfo, open(subjectinfo_path,'wb'))
    if len(contrasts) > 0:
        contrast_path = join(task_dir,'contrasts.pkl')
        pickle.dump(contrasts, open(contrast_path,'wb'))


# *********************************************
# ### Specify Input and Output Stream
# *********************************************

# Get Subject Info - get subject specific condition information
getsubjectinfo = Node(Function(input_names=['data_dir', 'subject_id', 'task',
                                            'use_events', 'regress_rt'],
                               output_names=['subjectinfo', 'contrasts'],
                               function=subjectinfo),
                      name='getsubjectinfo')
getsubjectinfo.inputs.data_dir = data_dir
getsubjectinfo.inputs.use_events = use_events
getsubjectinfo.inputs.regress_rt = regress_rt

# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'task',
                                            'contrasts']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('task', task_list)]
# SelectFiles - to grab the data (alternative to DataGrabber)
templates = {'func': join('*{subject_id}','*','func',
                         '*{task}*MNI*preproc.nii.gz'),
            'mask': join('*{subject_id}','*','func',
                         '*{task}*MNI*brainmask.nii.gz')}
selectfiles = Node(SelectFiles(templates,
                               base_directory = data_dir,
                               sort_filelist=True),
                   name="selectfiles")
# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory = output_dir,
                         container=first_level_dir),
                name="datasink")
# Save python objects that aren't accomodated by datasink nodes
save_subjectinfo = Node(Function(input_names=['base_directory','subject_id',
                                              'task','subjectinfo','contrasts'],
                                 output_names=['output_path'],
                                function=save_subjectinfo),
                       name="savesubjectinfo")
save_subjectinfo.inputs.base_directory = join(output_dir,first_level_dir)

# Use the following DataSink output substitutions
substitutions = [('_subject_id_', ''),
                ('fstat', 'FSTST'),
                ('run0.mat', 'designfile.mat')]
datasink.inputs.substitutions = substitutions

# *********************************************
# ### Model Specification
# *********************************************
# mask and blur
masker = Node(fsl.maths.ApplyMask(),name='masker')

# SpecifyModel - Generates FSL-specific Model
modelspec = Node(SpecifyModel(input_units='secs',
                              time_repetition=TR,
                              high_pass_filter_cutoff=80),
                 name="modelspec")
# Level1Design - Generates an FSL design matrix
level1design = Node(fsl.Level1Design(bases={'dgamma':{'derivs': True}},
                                     interscan_interval=TR,
                                     model_serial_correlations=True),
                        name="level1design")
# FEATmodel
level1model = Node(fsl.FEATModel(), name="FEATModel")

# FILMGLs
# smooth_autocorr, check default, use FSL default
filmgls = Node(fsl.FILMGLS(), name="FILMGLS")
# *********************************************
# # Workflow
# *********************************************

# Initiation of the 1st-level analysis workflow
l1analysis = Workflow(name='l1analysis')
l1analysis.base_dir = join(output_dir, working_dir)

# Connect up the 1st-level analysis components
l1analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                               ('task', 'task')]),
                    (infosource, getsubjectinfo, [('subject_id','subject_id'),
                                                 ('task', 'task')]),
                    (selectfiles, masker, [('func','in_file'),
                                           ('mask', 'mask_file')]),
                    (getsubjectinfo, modelspec, [('subjectinfo','subject_info')]),
                    (masker, modelspec, [('out_file', 'functional_runs')]),
                    (modelspec, level1design, [('session_info','session_info')]),
                    (getsubjectinfo, level1design, [('contrasts','contrasts')]),
                    (level1design, level1model, [('ev_files', 'ev_files'),
                                                 ('fsf_files','fsf_file')]),
                    (masker, filmgls, [('out_file', 'in_file')]),
                    (level1model, filmgls, [('design_file', 'design_file'),
                                            ('con_file', 'tcon_file'),
                                            ('fcon_file', 'fcon_file')]),
                    (level1model, datasink, [('design_file', '@design_file')]),
                    (filmgls, datasink, [('copes', '@copes'),
                                        ('zstats', '@Z'),
                                        ('fstats', '@F'),
                                        ('tstats','@T'),
                                        ('param_estimates','@param_estimates'),
                                        ('residual4d', '@residual4d'),
                                        ('sigmasquareds', '@sigmasquareds')]),
                    (getsubjectinfo, save_subjectinfo, [('subject_id','subject_id'),
                                                        ('task','tasj'),
                                                        ('subjectinfo','subjectinfo'),
                                                        ('contrasts','contrasts')])
                    
                    ])

l1analysis.run('MultiProc')


# workflow without events file
# reference https://nipype.readthedocs.io/en/latest/users/examples/rsfmri_vol_surface_preprocessing.html

"""
l1analysis.write_graph(graph2use='colored', format='png', simple_form=False)
graph_file=join(l1analysis.base_dir, 'l1analysis', 'graph.dot.png')
shutil.move(graph_file, join(output_dir, first_level_dir, 'graph.dot.png'))
if args.cleanup == True:
  shutil.rmtree(l1analysis.base_dir)
"""