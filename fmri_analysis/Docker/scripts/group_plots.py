import argparse
from glob import glob
from matplotlib import pyplot as plt
from nilearn import plotting
from os import path
from utils.display_utils import get_design_df, plot_contrasts, plot_design

# parse arguments
parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. These just consist of plots.')
parser.add_argument('--data_dir',help='The label(s) of the participant(s)'
                   'that should be analyzed. Multiple '
                   'participants can be specified with a space separated list.')
parser.add_argument('--tasks',help='The label(s) of the task(s)'
                   'that should be analyzed. If this parameter is not '
                   'provided all tasks should be analyzed.',
                   nargs="+")

args, unknown = parser.parse_known_args()
output_dir = args.output_dir

data_dir = '/Data' # /Data
if args.data_dir:
  data_dir = args.data_dir
  
  
# list of task identifiers
if args.tasks:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'discountFix', 'DPX', 'motorSelectiveStop',
               'stopSignal', 'stroop', 'twoByTwo']

data_dir = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/output/custom_modeling'
output_dir = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/output/Plots'

# plot tstat maps for each task
for task in tasks:
    tstat_files = glob(path.join(data_dir, '*%s*raw_tfile*' % task ))
    group_fig, group_axes = plt.subplots(len(tstat_files), 1,
                                     figsize=(14, 5*len(tstat_files)))
    for i, tfile in enumerate(tstat_files):
        basename = path.basename(tfile)
        title = basename[:(basename.find('raw')-1)]
        plotting.plot_stat_map(tfile, threshold=1, 
                               axes=group_axes[i],
                               title=title)
    group_fig.savefig(path.join(output_dir,'%s_raw_tfiles.png' % task))


# plot individual subject's contrasts and then the group
for task in tasks:
    # plot all group contrasts'
    plot_contrasts(data_dir, task, output_dir=output_dir, plot_individual=True)
    task_path = glob(path.join(data_dir,'*%s' % task))[0]
    design = get_design_df(task_path)
    plot_design(design, output_dir=path.join(output_dir,task))


