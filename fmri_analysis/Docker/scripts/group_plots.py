import argparse
from glob import glob
import json
from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn.image import iter_img
from os import makedirs, path
import pandas as pd
from utils.display_utils import dendroheatmap_left,get_design_df
from utils.display_utils import plot_contrasts, plot_design
import seaborn as sns

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

args = parser.parse_args()
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

# plot group map used
print('Plotting group mask...')
plotting.plot_roi(path.join(data_dir, 'group_mask.nii.gz'), 
                  output_file = path.join(output_dir,'group_mask.png'))

# plot tstat maps for each task
print('Plotting task contrasts...')
for task in tasks:
    task_dir = path.join(data_dir, task)
    subj_ids = json.load(open(path.join(task_dir,'subj_ids.json'),'r'))
    tstat_files = sorted(glob(path.join(task_dir, '*%s*raw_tfile*' % task )),
                         key = lambda x: '-' in x)
    group_fig, group_axes = plt.subplots(len(tstat_files), 1,
                                     figsize=(14, 6*len(tstat_files)))
    group_fig.suptitle('N = %s' % len(subj_ids), fontsize=30)
    plt.subplots_adjust(top=.95)
    for i, tfile in enumerate(tstat_files):
        basename = path.basename(tfile)
        title = basename[:(basename.find('raw')-1)]
        plotting.plot_stat_map(tfile, threshold=1, 
                               axes=group_axes[i],
                               title=title)
    makedirs(path.join(output_dir,task), exist_ok=True)
    group_fig.savefig(path.join(output_dir,task,'%s_raw_tfiles.png' % task))

# Plot ICA components
print('Plotting ICA...')
for n_comps in [20, 40]:
    print('Plotting n components: %s' % n_comps)
    components_img = path.join(data_dir, 
                               'canica%s_explicit_contrasts.nii.gz' % n_comps)
    # plot all components in one map
    plotting.plot_prob_atlas(components_img, title='All %s ICA components' % n_comps,
                             output_file = path.join(output_dir,
                                                   'canica%s_allcomps.png' % \
                                                     n_comps))
    # plot each component separately
    ica_fig, ica_axes = plt.subplots(n_comps//4, 4, figsize=(14, n_comps//4*5))
    ica_fig.suptitle('CanICA - 20 Components')
    for i, cur_img in enumerate(iter_img(components_img)):
        plotting.plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=1, colorbar=False, axes = ica_fig.axes[i])
    ica_fig.savefig(path.join(output_dir, 'canica%s_sep_comps.png' % n_comps))

    # Plot projection onto ICA components
    print('Plotting projection...')
    projection = pd.read_json(path.join(data_dir, 
                                        'canica%s_projection.json' % n_comps))
    cluster_map = dendroheatmap_left(projection.T.corr(), label_fontsize=6)
    cluster_map[0].savefig(path.join(output_dir, 'canica%s_projection_dendroheatmap.png' % n_comps))

# plot individual subject's contrasts and then the group
print('Plotting individual beta maps...')
for task in tasks:
    # plot all group contrasts'
    plot_contrasts(data_dir, task, output_dir=output_dir, plot_individual=True)
