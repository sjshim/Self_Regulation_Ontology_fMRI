
# coding: utf-8

# ### Imports

# In[ ]:


import argparse
from inspect import currentframe, getframeinfo
from glob import glob
from pathlib import Path
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.base import Bunch
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
import os
from os.path import join
import pandas as pd
import pickle
import sys
from utils.event_utils import get_beta_series, get_contrasts, parse_EVs, process_confounds


# ### Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute task_analysis.ipynb

# In[ ]:


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('-derivatives_dir', default='/derivatives')
parser.add_argument('-data_dir', default='/data')
parser.add_argument('--participant_label')
parser.add_argument('--tasks', nargs="+")
parser.add_argument('--skip_beta', action='store_false')
parser.add_argument('--skip_contrast', action='store_false')
parser.add_argument('--n_procs', default=16)
if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/mnt/OAK/derivatives'
    args.data_dir = '/mnt/OAK'
    args.tasks = ['stroop']
    args.participant_label = 's611'
    args.n_procs=4


# ### Initial Setup

# In[ ]:


# get current directory to pass to function nodes
filename = getframeinfo(currentframe()).filename
current_directory = str(Path(filename).resolve().parent)

# list of subject identifiers
subject_id = args.participant_label
# list of task identifiers
if args.tasks is not None:
    task_list = args.tasks
else:
    task_list = ['ANT', 'CCTHot', 'discountFix',
               'DPX', 'motorSelectiveStop',
               'stopSignal', 'stroop', 'surveyMedley',
               'twoByTwo', 'WATT3']

#### Experiment Variables
derivatives_dir = args.derivatives_dir
fmriprep_dir = join(derivatives_dir, 'fmriprep', 'fmriprep')
data_dir = args.data_dir
first_level_dir = join(derivatives_dir,'1stLevel')
working_dir = 'workingdir'
run_beta = args.skip_beta
run_contrast = args.skip_contrast
n_procs = args.n_procs
# TR of functional images
TR = .68


# In[ ]:


# print
print('*'*79)
print('Task List: %s\n, Subject: %s\n, derivatives_dir: %s\n, data_dir: %s' % 
     (task_list, subject_id, derivatives_dir, data_dir))
print('Running Contrast?: %s, Running Beta?: %s' % 
     (['No','Yes'][run_contrast], ['No','Yes'][run_beta]))
print('*'*79)


# # Set up Nodes

# ### Define helper functions

# In[ ]:


def get_events_regressors(data_dir, fmirprep_dir, subject_id, task):
    # strip "sub" from beginning of subject_id if provided
    subject_id = subject_id.replace('sub-','')
    ## Get the Confounds File (output of fmriprep)
    # Read the TSV file and convert to pandas dataframe
    confounds_file = glob(join(fmriprep_dir,
                               'sub-%s' % subject_id,
                               '*', 'func',
                               '*%s*confounds.tsv' % task))[0]
    regressors, regressor_names = process_confounds(confounds_file)
    ## Get the Events File if it exists
    # Read the TSV file and convert to pandas dataframe
    event_file = glob(join(data_dir,
                           'sub-%s' % subject_id,
                           '*', 'func',
                           '*%s*events.tsv' % task))   
    if len(event_file)>0:
        # set up events file
        event_file = event_file[0]
        events_df = pd.read_csv(event_file,sep = '\t')
    else:
        events_df = None
    regressors, regressor_names = process_confounds(confounds_file)
    return events_df, regressors, regressor_names

# helper function to create bunch
def getsubjectinfo(events_dr, regressors, regressor_names, task='beta', regress_rt=True): 
    EV_dict = parse_EVs(events_df, task, regress_rt)
    contrasts = []
    if task not in ['beta']:
        contrasts = get_contrasts(task, regress_rt)
    # create beta series info
    subjectinfo = Bunch(conditions=EV_dict['conditions'],
                        onsets=EV_dict['onsets'],
                        durations=EV_dict['durations'],
                        amplitudes=EV_dict['amplitudes'],
                        tmod=None,
                        pmod=None,
                        regressor_names=regressor_names,
                        regressors=regressors.T.tolist(),
                        contrasts=contrasts)
    return subjectinfo
    
def save_subjectinfo(save_directory, subjectinfo):
    os.makedirs(save_directory, exist_ok=True)
    subjectinfo_path = join(save_directory, 'subjectinfo.pkl')
    pickle.dump(subjectinfo, open(subjectinfo_path,'wb'))


# ### Specify Input and Output Stream

# In[ ]:


def get_selector(task, subject_id, session=None):
    if session is None:
        ses = '*'
    else:
        ses = 'ses-%s' % str(session)
    # SelectFiles - to grab the data (alternative to DataGrabber)
    templates = {'func': join('sub-{subject_id}',ses,'func',
                             '*{task}*MNI*preproc.nii.gz'),
                 'mask': join('sub-{subject_id}',ses,'func',
                              '*{task}*MNI*brainmask.nii.gz')}
    selectfiles = Node(SelectFiles(templates,
                                   base_directory=fmriprep_dir,
                                   sort_filelist=True),
                       name='%s_selectFiles' % task)
    selectfiles.inputs.task = task
    selectfiles.inputs.subject_id = subject_id
    return selectfiles

def get_masker(name):
    # mask and blur
    return Node(fsl.maths.ApplyMask(),name=name)


# # Create workflow

# ### helper functions

# In[ ]:


def init_common_wf(workflow, task):
    # initiate basic nodes
    masker = get_masker('%s_masker' % task)
    selectfiles = get_selector(task, subject_id)
    # Connect up the 1st-level analysis components
    workflow.connect([(selectfiles, masker, [('func','in_file'), ('mask', 'mask_file')])])

def init_GLM_wf(subject_info, task, name='model-standard_wf-standard', contrasts=None):
    # Datasink - creates output folder for important outputs
    datasink = Node(DataSink(base_directory=first_level_dir,
                             container=subject_id), name="datasink")
    # Use the following DataSink output substitutions
    substitutions = [('_subject_id_', ''),
                    ('fstat', 'FSTST'),
                    ('run0.mat', 'designfile.mat')]
    
    datasink.inputs.substitutions = substitutions
    # ridiculous regexp substitution to get files just right
    # link to ridiculousness: https://regex101.com/r/ljS5zK/3
    match_str = "(?P<sub>s[0-9]+)\/(?P<task>[A-Za-z1-9_]+)_(?P<model>model-[a-z]+)_(?P<submodel>wf-[a-z]+)\/(s[0-9]+/|)"
    replace_str = "\g<sub>/\g<task>/\g<model>/\g<submodel>/"
    regexp_substitutions = [(match_str, replace_str)]
    datasink.inputs.regexp_substitutions = regexp_substitutions
    
    # SpecifyModel - Generates FSL-specific Model
    modelspec = Node(SpecifyModel(input_units='secs',
                                  time_repetition=TR,
                                  high_pass_filter_cutoff=80),
                     name="%s_modelspec" % task)
    modelspec.inputs.subject_info = subject_info
    # Level1Design - Creates FSL config file 
    level1design = Node(fsl.Level1Design(bases={'dgamma':{'derivs': True}},
                                         interscan_interval=TR,
                                         model_serial_correlations=True),
                            name="%s_level1design" % task)
    level1design.inputs.contrasts=subject_info.contrasts
    # FEATmodel generates an FSL design matrix
    level1model = Node(fsl.FEATModel(), name="%s_FEATModel" % task)

    # FILMGLs
    # smooth_autocorr, check default, use FSL default
    filmgls = Node(fsl.FILMGLS(), name="%s_GLS" % task)

    wf = Workflow(name='%s_%s' % (task,name))
    wf.connect([(modelspec, level1design, [('session_info','session_info')]),
                (level1design, level1model, [('ev_files', 'ev_files'),
                                             ('fsf_files','fsf_file')]),
                (level1model, datasink, [('design_file', '%s.@design_file' % name)]),
                (level1model, filmgls, [('design_file', 'design_file'),
                                        ('con_file', 'tcon_file'),
                                        ('fcon_file', 'fcon_file')]),
                (filmgls, datasink, [('copes', '%s.@copes' % name),
                                     ('zstats', '%s.@Z' % name),
                                     ('fstats', '%s.@F' % name),
                                     ('tstats','%s.@T' % name),
                                     ('param_estimates','%s.@param_estimates' % name),
                                     ('residual4d', '%s.@residual4d' % name),
                                     ('sigmasquareds', '%s.@sigmasquareds' % name)])
               ])
    return wf



def get_task_wfs(task, beta_subjectinfo=None, contrast_subjectinfo=None, regress_rt=True):
    rt_suffix = 'rt' if regress_rt==True else 'nort'
    # set up workflow lookup
    wf_dict = {'contrast': (init_GLM_wf, {'name': 'model-%s_wf-contrast' % rt_suffix,
                                          'task': task}), 
               'beta': (init_GLM_wf, {'name': 'model-%s_wf-beta' % rt_suffix,
                                      'task': task})}
    
    workflows = []
    if beta_subjectinfo:
        save_directory = join(first_level_dir, subject_id, task, 'model-%s' % rt_suffix, 'wf-beta')
        save_subjectinfo(save_directory, beta_subjectinfo)
        func, kwargs = wf_dict['beta']
        workflows.append(func(beta_subjectinfo, **kwargs))
    if contrast_subjectinfo:
        save_directory = join(first_level_dir, subject_id, task, 'model-%s' % rt_suffix, 'wf-contrast')
        save_subjectinfo(save_directory, contrast_subjectinfo)
        func, kwargs = wf_dict['contrast']
        workflows.append(func(contrast_subjectinfo, **kwargs))
    return workflows
    


# In[ ]:


# Initiation of the 1st-level analysis workflow
l1analysis = Workflow(name='%s_l1analysis' % subject_id)
l1analysis.base_dir = join(derivatives_dir, working_dir)

for task in task_list:
    init_common_wf(l1analysis, task)
    # get nodes to pass
    masker = l1analysis.get_node('%s_masker' % task)
    # get info to pass to task workflows
    events_df, regressors, regressor_names = get_events_regressors(data_dir, fmriprep_dir,
                                                                   subject_id, task)
    # perform analyses both by regressing rt and not
    regress_rt_conditions = [True, False]
    if 'stop' in task:
        regress_rt_conditions = [False]
    betainfo = None; contrastinfo = None
    for regress_rt in regress_rt_conditions:
        if run_beta:
            betainfo = getsubjectinfo(events_df, regressors, regressor_names, task='beta', regress_rt=regress_rt)
        if run_contrast:
            contrastinfo = getsubjectinfo(events_df, regressors, regressor_names, task=task, regress_rt=regress_rt)
        task_workflows = get_task_wfs(task, betainfo, contrastinfo, regress_rt)
        for wf in task_workflows:
            l1analysis.connect([
                                (masker, wf, [('out_file', '%s_modelspec.functional_runs' % task)]),
                                (masker, wf, [('out_file','%s_GLS.in_file' % task)])
                                ])
        


# ### Run the Workflow
# 

# In[ ]:


#l1analysis.run()
l1analysis.run('MultiProc', plugin_args={'n_procs': n_procs})

