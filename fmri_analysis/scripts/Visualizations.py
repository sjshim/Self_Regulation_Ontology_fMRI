
# coding: utf-8

# In[ ]:

import argparse
from glob import glob
import os
from os import path
import pickle
from nilearn import plotting
from nistats.thresholding import map_threshold
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

from utils.firstlevel_utils import get_first_level_objs
from utils.firstlevel_plot_utils import (plot_design, plot_design_timeseries, 
                                         plot_design_heatmap, plot_contrast,
                                        plot_map, plot_task_maps)


# In[ ]:

parser = argparse.ArgumentParser(description='2nd level Entrypoint Script.')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('--skip_designs', action='store_true')
parser.add_argument('--skip_first', action='store_true')
parser.add_argument('--skip_second', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix,                                     DPX, motorSelectiveStop, stopSignal,                                     stroop, surveyMedley, twoByTwo, WATT3")

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/mnt/OAK/data/uh2/BIDS_data/derivatives/'
    args.data_dir = '/mnt/OAK/data/uh2/BIDS_data/'
    args.tasks = ['ANT', 'CCTHot', 'discountFix',
        'DPX', 'motorSelectiveStop',
        'stopSignal', 'stroop',
        'twoByTwo', 'WATT3']
    # args.rt=True
    args.save=True
    get_ipython().magic(u'matplotlib inline')


# In[ ]:

# set paths
first_level_dir = path.join(args.derivatives_dir, '1stlevel')
print(first_level_dir)
second_level_dir = path.join(args.derivatives_dir,'2ndlevel')
fmriprep_dir = path.join(args.derivatives_dir, 'fmriprep', 'fmriprep')
tasks = args.tasks
# tasks = ['ANT', 'CCTHot', 'discountFix',
#         'DPX', 'motorSelectiveStop',
#         'stopSignal', 'stroop',
#         'twoByTwo', 'WATT3']
save = args.save
plot_designs = not args.skip_designs
run_first_level = not args.skip_first
run_second_level = not args.skip_second


# # Design Visualization

# In[ ]:

# load design
# subject_id, task = 's592', 'stroop'
# files = get_first_level_objs(subject_id, task, first_level_dir, regress_rt=False)
# subjinfo = pickle.load(open(files[0], 'rb'))


# In[ ]:



# display the glm to make sure its not wonky - useful for doublechecking, not necessary #
#    from nistats.reporting import plot_design_matrix
#    import matplotlib.pyplot as plt
#    fig, (ax1) = plt.subplots(figsize=(20, 10), nrows=1, ncols=1)
#    ax1.set_title(subjinfo)
#    plot_design_matrix(design_matrix, ax=ax1)
####


# In[ ]:
if plot_designs:
    rcParams.update({'figure.autolayout': True})
    subjects = sorted([i.split("/")[-1] for i in glob(os.path.join(first_level_dir, '*')) if 'fig' not in i])
    print(subjects)
    for sub in subjects: 
        for task in tasks:
            print(sub, task)
            try:
                files = get_first_level_objs(sub, task, first_level_dir, regress_rt=True, beta = False)[0]
                print(files)
                f = open(files, 'rb')
                subjinfo = pickle.load(f)
                f.close()

                plot_design(subjinfo)
                plt.gcf()
                plt.savefig('%s/figs/%s_%s_design_fig' % (first_level_dir, sub, task))
                plt.close()

                plot_design_timeseries(subjinfo)
                plt.gcf()
                plt.savefig('%s/figs/%s_%s__design_timeseries' % (first_level_dir, sub, task))
                plt.close()

                if task=='WATT3':
                    plot_design_timeseries(subjinfo, 0, 200)
                    plt.gcf()
                    plt.savefig('%s/figs/%s_%s__design_timeseries_trunc' % (first_level_dir, sub, task))
                    plt.close()
                else:
                    plot_design_timeseries(subjinfo, 0, 100)
                    plt.gcf()
                    plt.savefig('%s/figs/%s_%s__design_timeseries_trunc' % (first_level_dir, sub, task))
                    plt.close()                


        
                plot_design_heatmap(subjinfo)
                plt.gcf()
                plt.savefig('%s/figs/%s_%s_heatmap' % (first_level_dir, sub, task), bbox_inches="tight")
                plt.close()
            except:
                print('task not found')

        


# # First Level Visualization

# In[ ]:

if run_first_level:
    for task in tasks:
        contrast_maps = glob(path.join(first_level_dir, '*', task, '*maps*', '*.nii.gz'))
        for map_file in contrast_maps:
            contrast_name = map_file[map_file.index('contrast')+9:].rstrip('.nii.gz')
            f = plot_map(map_file, title=contrast_name)
            if save:
                output = map_file.replace('.nii.gz', '_plots.pdf')
                f.savefig(output)


# # Second Level Visualization

# In[ ]:

# if run_second_level:
#     for task in tasks:
#         contrast_maps = sorted(glob(path.join(second_level_dir, task, '*maps', '*.nii.gz')))
#         for map_file in contrast_maps:
#             contrast_name = map_file[map_file.index('contrast')+9:].rstrip('.nii.gz')
#             # plot
#             f = plot_map(map_file, title=contrast_name)
#             if save:
#                 output = map_file.replace('.nii.gz', '_plots.pdf')
#                 f.savefig(output)

if run_second_level:
    for task in tasks:
        contrast_dirs = sorted(glob(path.join(second_level_dir, task, '*maps')))
        for contrast_dir in contrast_dirs:
            contrast_maps = sorted(glob(path.join(contrast_dir, '*.nii.gz')))
            beta_maps = [mapi for mapi in contrast_maps if 'tfile' not in mapi] 
            t_maps = [mapi for mapi in contrast_maps if 'raw_tfile' in mapi] 
            corrected_t_maps = [mapi for mapi in contrast_maps if 'corrected_tfile' in mapi] 
            task_name = contrast_maps[0].split('/2ndlevel/')[-1].split('/')[0]
            RT_flag = 'RT-True' in contrast_dir
            curr_title = task_name+'_RT-'+str(RT_flag)
            
            f_beta = plot_task_maps(beta_maps, curr_title)
            f_raw_t = plot_task_maps(t_maps, curr_title)
            f_corr_t = plot_task_maps(corrected_t_maps, curr_title)
            if save:
                output_beta = path.join(contrast_dir, task_name+'_plots.pdf')
                f_beta.savefig(output_beta)
                output_raw_t = path.join(contrast_dir, task_name+'_raw_tfile_plots.pdf')
                f_beta.savefig(output_raw_t)
                output_corr_t = path.join(contrast_dir, task_name+'_corrected_tfile_plots.pdf')
                f_beta.savefig(output_corr_t)  
