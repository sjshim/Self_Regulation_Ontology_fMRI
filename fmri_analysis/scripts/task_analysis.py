
# coding: utf-8

# ### Imports

# In[1]:


import argparse
from inspect import currentframe, getframeinfo
from pathlib import Path
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
import os
from os.path import join
import sys
from utils.event_utils import get_contrasts


# ### Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute task_analysis.ipynb

# In[2]:


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('-derivatives_dir', default='/derivatives')
parser.add_argument('-data_dir', default='/data')
parser.add_argument('--participant_labels',nargs="+")
parser.add_argument('--tasks', nargs="+")
parser.add_argument('--ignore_rt', action='store_true')
parser.add_argument('--cleanup', action='store_true')
if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/mnt/OAK/derivatives'
    args.data_dir = '/mnt/OAK'
    args.tasks = ['stroop']
    args.participant_labels = ['s130']


# ### Initial Setup

# In[3]:


# get current directory to pass to function nodes
filename = getframeinfo(currentframe()).filename
current_directory = str(Path(filename).resolve().parent)

# list of subject identifiers
subject_list = args.participant_labels
# list of task identifiers
if args.tasks is not None:
    task_list = args.tasks
else:
    task_list = ['ANT', 'CCTHot', 'discountFix',
               'DPX', 'motorSelectiveStop',
               'stopSignal', 'stroop', 'surveyMedley',
               'twoByTwo', 'WATT3']

regress_rt = not args.ignore_rt
if regress_rt:
    rt_suffix = 'rt'
else:
    rt_suffix = 'nort'
#### Experiment Variables
derivatives_dir = args.derivatives_dir
fmriprep_dir = join(derivatives_dir, 'fmriprep', 'fmriprep')
data_dir = args.data_dir
first_level_dir = join(derivatives_dir,'1stLevel')
working_dir = 'workingdir'
# TR of functional images
TR = .68


# In[4]:


# print
print('*'*79)
print('Task List: %s\n, Subjects: %s\n, derivatives_dir: %s\n, data_dir: %s' % 
     (task_list, subject_list, derivatives_dir, data_dir))
print('*'*79)


# # Set up Nodes

# ### Define helper functions

# In[5]:


# helper function to create bunch
def getsubjectinfo(data_dir, fmriprep_dir, subject_id, task, regress_rt, utils_path): 
    from glob import glob
    from os.path import join
    import pandas as pd
    from nipype.interfaces.base import Bunch
    import sys
    sys.path.append(utils_path)
    from utils.event_utils import get_beta_series, parse_EVs, process_confounds
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
    beta_subjectinfo = None
    contrast_subjectinfo = None
    contrast = None
    if len(event_file)>0:
        # set up events file
        event_file = event_file[0]
        events_df = pd.read_csv(event_file,sep = '\t')
        EV_dict, beta_dict = parse_EVs(events_df, task, regress_rt, beta=True)
        # create beta series info
        beta_subjectinfo = Bunch(subject_id=subject_id,
                                 task=task,
                                 conditions=beta_dict['conditions'],
                                 onsets=beta_dict['onsets'],
                                 durations=beta_dict['durations'],
                                 amplitudes=beta_dict['amplitudes'],
                                 tmod=None,
                                 pmod=None,
                                 regressor_names=regressor_names,
                                 regressors=regressors.T.tolist())
        # set up contrasts
        contrast_subjectinfo = Bunch(subject_id=subject_id,
                                     task=task,
                                     conditions=EV_dict['conditions'],
                                     onsets=EV_dict['onsets'],
                                     durations=EV_dict['durations'],
                                     amplitudes=EV_dict['amplitudes'],
                                     tmod=None,
                                     pmod=None,
                                     regressor_names=regressor_names,
                                     regressors=regressors.T.tolist())
    return beta_subjectinfo, contrast_subjectinfo
    
def save_subjectinfo(save_directory, beta_subjectinfo, contrast_subjectinfo, contrasts=[], model_name='standard'):
    from os import makedirs
    from os.path import join
    import pickle
    subject_id = beta_subjectinfo.subject_id
    task = beta_subjectinfo.task
    subjectinfo_dir = join(save_directory, subject_id, task, 'model-%s' % model_name)
    makedirs(subjectinfo_dir, exist_ok=True)
    # save beta subject info
    makedirs(join(subjectinfo_dir, 'wf-beta'), exist_ok=True)
    beta_path = join(subjectinfo_dir, 'wf-beta', 'subjectinfo.pkl')
    pickle.dump(beta_subjectinfo, open(beta_path,'wb'))
    # save contrast subject info
    if len(contrast_subjectinfo.items()) > 0:
        makedirs(join(subjectinfo_dir, 'wf-contrast'), exist_ok=True)
        contrast_path = join(subjectinfo_dir, 'wf-contrast', 'subjectinfo.pkl')
        pickle.dump(contrast_subjectinfo, open(contrast_path,'wb'))
        # save contrast list
        contrastlist_path = join(subjectinfo_dir,'wf-contrast', 'contrasts.pkl')
        pickle.dump(contrasts, open(contrastlist_path,'wb'))


# View one events file used in subject info

# ### Specify Input and Output Stream

# In[6]:


def get_subjectinfo(name):
    # Get Subject Info - get subject specific condition information
    subjectinfo = Node(Function(input_names=['data_dir', 'fmriprep_dir','subject_id', 
                                             'task','regress_rt', 'utils_path'],
                                   output_names=['beta_subjectinfo', 
                                                 'contrast_subjectinfo'],
                                   function=getsubjectinfo),
                          name=name)
    subjectinfo.inputs.fmriprep_dir = fmriprep_dir
    subjectinfo.inputs.data_dir = data_dir
    subjectinfo.inputs.regress_rt = regress_rt
    subjectinfo.inputs.utils_path = current_directory
    return subjectinfo

def get_savesubjectinfo(name):
    # Save python objects that aren't accomodated by datasink nodes
    savesubjectinfo = Node(Function(input_names=['save_directory',
                                                 'beta_subjectinfo',
                                                 'contrast_subjectinfo',
                                                 'contrasts',
                                                 'model_name'],
                                    function=save_subjectinfo),
                           name=name)
    savesubjectinfo.inputs.save_directory = first_level_dir
    savesubjectinfo.inputs.model_name = rt_suffix
    return savesubjectinfo

def get_selector(name, session=None):
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
                       name=name)
    return selectfiles

def get_masker(name):
    # mask and blur
    return Node(fsl.maths.ApplyMask(),name=name)


# # Create workflow

# ### helper functions

# In[7]:


def init_common_wf(workflow, task):
        # Infosource - a function free node to iterate over the list of subject names
    infosource = Node(IdentityInterface(fields=['subject_id',
                                                'task',
                                                'contrasts']),
                      name="%s_infosource" % task)
    infosource.iterables = [('subject_id', subject_list)]
    infosource.inputs.task = task
    infosource.inputs.contrasts = get_contrasts(task, regress_rt)
    
    # initiate basic nodes
    subjectinfo = get_subjectinfo('%s_subjectinfo' % task)
    savesubjectinfo = get_savesubjectinfo('%s_savesubjectinfo' % task)
    masker = get_masker('%s_masker' % task)
    selectfiles = get_selector('%s_selectFiles' % task)
    
    # Connect up the 1st-level analysis components
    workflow.connect([(infosource, selectfiles, [('subject_id', 'subject_id'), ('task', 'task')]),
                      (infosource, subjectinfo, [('subject_id','subject_id'), ('task', 'task')]),
                      (infosource, savesubjectinfo, [('contrasts','contrasts')]),
                      (subjectinfo, savesubjectinfo, [('beta_subjectinfo','beta_subjectinfo'),
                                                      ('contrast_subjectinfo','contrast_subjectinfo')]),
                      (selectfiles, masker, [('func','in_file'),
                                             ('mask', 'mask_file')])
                        ])

def init_GLM_wf(name='wf-standard'):
    # Datasink - creates output folder for important outputs
    datasink = Node(DataSink(base_directory=first_level_dir), name="datasink")
    # Use the following DataSink output substitutions
    substitutions = [('_subject_id_', ''),
                    ('fstat', 'FSTST'),
                    ('run0.mat', 'designfile.mat')]
    
    datasink.inputs.substitutions = substitutions
    # ridiculous regexp substitution to get files just right
    # link to ridiculousness: https://regex101.com/r/ljS5zK/1
    match_str = "(?P<sub>s[0-9]+)\/(?P<task>[a-z_]+)_(?P<model>model-[a-z]+)_(?P<submodel>wf-[a-z_]+)\/s[0-9]+"
    replace_str = "\g<sub>/\g<task>/\g<model>/\g<submodel>"
    regexp_substitutions = [(match_str, replace_str)]
    datasink.inputs.regexp_substitutions = regexp_substitutions
    
    # SpecifyModel - Generates FSL-specific Model
    modelspec = Node(SpecifyModel(input_units='secs',
                                  time_repetition=TR,
                                  high_pass_filter_cutoff=80),
                     name="modelspec")
    # Level1Design - Creates FSL config file 
    level1design = Node(fsl.Level1Design(bases={'dgamma':{'derivs': True}},
                                         interscan_interval=TR,
                                         model_serial_correlations=True),
                            name="level1design")
    # FEATmodel generates an FSL design matrix
    level1model = Node(fsl.FEATModel(), name="FEATModel")

    # FILMGLs
    # smooth_autocorr, check default, use FSL default
    filmgls = Node(fsl.FILMGLS(), name="GLS")

    wf = Workflow(name=name)
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



def get_task_wfs(task):
    wf_dict = {}
    # set up workflow lookup
    default_wf = [(init_GLM_wf, {'name': '%s_model-%s_wf-contrast' % (task, rt_suffix)}), 
                  (init_GLM_wf, {'name': '%s_model-%s_wf-beta' % (task, rt_suffix)})]
    
    # get workflow
    workflows = []
    for func, kwargs in wf_dict.get(task, default_wf):
        workflows.append(func(**kwargs))
    return workflows
    


# In[8]:


# Initiation of the 1st-level analysis workflow
l1analysis = Workflow(name='l1analysis')
l1analysis.base_dir = join(derivatives_dir, working_dir)

for task in task_list:
    init_common_wf(l1analysis, task)
    task_workflows = get_task_wfs(task)
    # get nodes to pass
    infosource = l1analysis.get_node('%s_infosource' % task)
    subjectinfo = l1analysis.get_node('%s_subjectinfo' % task)
    masker = l1analysis.get_node('%s_masker' % task)
    for wf in task_workflows:
        l1analysis.connect([
                            (infosource, wf, [('subject_id','datasink.container')]),
                            (subjectinfo, wf, [('contrast_subjectinfo','modelspec.subject_info')]),
                            (masker, wf, [('out_file', 'modelspec.functional_runs')]),
                            (masker, wf, [('out_file','GLS.in_file')])
                            ])
        
        if 'contrast' in wf.name:
            l1analysis.connect([(infosource, wf, [('contrasts','level1design.contrasts')])])


# ### Run the Workflow
# 

# In[9]:


#l1analysis.run()
l1analysis.run('MultiProc', plugin_args={'n_procs': 8})

