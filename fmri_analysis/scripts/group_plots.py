import argparse
from glob import glob
import json
from matplotlib import pyplot as plt
from nilearn import datasets, image
from nilearn import plotting
from nilearn.image import iter_img
from os import makedirs, path
import pandas as pd
from utils.utils import projections_corr
from utils.display_utils import dendroheatmap_left, plot_contrasts
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
               'stopSignal', 'stroop', 'twoByTwo', 'WATT3']

# plot group map used
print('Plotting group mask...')
plotting.plot_roi(path.join(data_dir, 'group_mask.nii.gz'), 
                  output_file = path.join(output_dir,'group_mask.png'))

# plot tstat maps for each task
print('Plotting task contrasts...')
for task in tasks:
    for tfile in ['raw', 'correct']:
        task_dir = path.join(data_dir, task)
        subj_ids = json.load(open(path.join(task_dir,'subj_ids.json'),'r'))
        tstat_files = sorted(glob(path.join(task_dir, '*%s*%s_tfile*' % (task, tfile))),
                             key = lambda x: '-' in x)
        group_fig, group_axes = plt.subplots(len(tstat_files), 1,
                                         figsize=(14, 6*len(tstat_files)))
        group_fig.suptitle('N = %s' % len(subj_ids), fontsize=30)
        plt.subplots_adjust(top=.95)
        for i, tfile in enumerate(tstat_files):
            basename = path.basename(tfile)
            title = basename[:(basename.find('raw')-1)]
            plotting.plot_stat_map(tfile, threshold=2, 
                                   axes=group_axes[i],
                                   title=title)
        makedirs(path.join(output_dir,task), exist_ok=True)
        group_fig.savefig(path.join(output_dir,task,'%s_%s_tfiles.png' % (task, tfile)))

# Plot ICA components
print('Plotting ICA...')
smith_networks = datasets.fetch_atlas_smith_2009()['rsn70']
parcellation_files = [('smith70', smith_networks),
                      ('canica20', 
                       path.join(data_dir, 'canica20_explicit_contrasts.nii.gz')),
                      ('canica50', 
                       path.join(data_dir, 'canica50_explicit_contrasts.nii.gz')),
                      ('canica70', 
                       path.join(output_dir, 'canica70_explicit_contrasts.nii.gz'))
                       ]
        
        
for parcellation_name, parcellation_file in parcellation_files:
    print('Plotting parcellation: %s' % parcellation_name)
    # plot all components in one map
    parcellation = image.load_img(parcellation_file)
    n_comps = parcellation.shape[-1]
    plotting.plot_prob_atlas(parcellation_file, title='%s ICA components' % parcellation_name,
                             output_file = path.join(output_dir,
                                                   '%s_allcomps.png' % \
                                                     parcellation_name))
    # plot each component separately
    ica_fig, ica_axes = plt.subplots(n_comps//4, 4, figsize=(14, n_comps//4*5))
    ica_fig.suptitle('%s - %s Components' % (parcellation_name, n_comps))
    for i, cur_img in enumerate(iter_img(parcellation_file)):
        plotting.plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=1, colorbar=False, axes = ica_fig.axes[i])
    ica_fig.savefig(path.join(output_dir, '%s_sep_comps.png' % parcellation_name))

    # Plot projection onto ICA components
    print('Plotting projection...')
    projections_df = pd.read_json(path.join(data_dir, 
                                        '%s_projection.json' % parcellation_name))
    corr = projections_corr(projections_df)
    cluster_map = dendroheatmap_left(corr, labels=False)
    labels = list(projections_df.index[cluster_map[1]])
    cluster_map[0].savefig(path.join(output_dir, '%s_projection_dendroheatmap.png' % parcellation_name))
    json.dump(labels, open(path.join(output_dir, '%s_projection_dendroheatmap_labels.json' % parcellation_name),'w'))
    
    # plot averages
    for group in ['subj','contrast']:
        avg_corr = projections_corr(projections_df, grouping=group)
        fig = plt.figure(figsize=(14,14))
        sns.heatmap(avg_corr, square=True)
        plt.tight_layout()
        fig.savefig(path.join(output_dir, 'projection_%s_avgcorr_%s_heatmap.png' % (parcellation_name, group)))

# plot individual subject's contrasts and then the group
print('Plotting individual beta maps...')
for task in tasks:
    # plot all group contrasts'
    plot_contrasts(data_dir, task, output_dir=output_dir, plot_individual=True)
