import argparse
from glob import glob
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

data_dir = '/mnt/Sherlock_Scratch/datasink/1stLevel/' # /Data
if args.data_dir:
  data_dir = args.data_dir
  
  
# list of task identifiers
if args.tasks:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'DPX', 'motorSelectiveStop',
               'stopSignal', 'stroop', 'twoByTwo']

# plot individual subject's contrasts and then the group
for task in tasks:
    # plot all group contrasts'
    plot_contrasts(data_dir, task, output_dir=output_dir, plot_individual=False)

