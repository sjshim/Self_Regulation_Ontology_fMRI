
# coding: utf-8

# In[ ]:

import argparse
from glob import glob
import os
import json
import re
from os import makedirs, path
import pickle
import nibabel as nib
import numpy as np
from nilearn import plotting
from nistats.thresholding import map_threshold
from nilearn.image import math_img
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

from utils.firstlevel_utils import get_first_level_objs
from utils.plot_utils import (plot_design, plot_design_timeseries,
                              plot_design_heatmap, plot_contrast,
                              plot_map, plot_task_maps,
                              get_contrast_title,
                              plot_vif)


# In[ ]:

parser = argparse.ArgumentParser(description='2nd level Entrypoint Script.')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('--skip_designs', action='store_true')
parser.add_argument('--skip_first', action='store_true')
parser.add_argument('--skip_second', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--tasks', nargs="+", default='ANT, CCTHot, discountFix, DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3'.split(', '), help="Choose from ANT, CCTHot, discountFix, DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('-group', default='NONE')

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
all_tasks = ['ANT', 'CCTHot', 'discountFix', 'DPX', 'motorSelectiveStop', 'stopSignal', 'stopSignal', 'twoByTwo', 'WATT3']
first_level_dir = path.join(args.derivatives_dir, '1stlevel')
print(first_level_dir)
second_level_dir = path.join(args.derivatives_dir,'2ndlevel')
fmriprep_dir = path.join(args.derivatives_dir, 'fmriprep', 'fmriprep')
tasks = all_tasks if 'all' in args.tasks else args.tasks
# tasks = ['ANT', 'CCTHot', 'discountFix',
#         'DPX', 'motorSelectiveStop',
#         'stopSignal', 'stroop',
#         'twoByTwo', 'WATT3']
save = args.save
plot_designs = not args.skip_designs
run_first_level = not args.skip_first
run_second_level = not args.skip_second
group = args.group



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
    fig_dir = os.path.join(first_level_dir, 'figs')
    makedirs(fig_dir, exist_ok=True)
    subjects = sorted([i.split("/")[-1] for i in glob(os.path.join(first_level_dir, '*')) if 'fig' not in i])
    print(subjects)
    for sub in subjects: 
        for task in tasks:
            print(sub, task)
            try:
                files = get_first_level_objs(sub, task, first_level_dir, regress_rt=True, beta = False)[0]
                print(files)
                with open(files, 'rb') as f:
                    subjinfo = pickle.load(f)
                
                plot_design(subjinfo)
                plt.gcf()
                plt.savefig(os.path.join(fig_dir, '%s_%s_design_fig' % (sub, task)))
                plt.close()

                plot_design_timeseries(subjinfo)
                plt.gcf()
                plt.savefig(os.path.join(fig_dir, '%s_%s__design_timeseries' % (sub, task)))
                plt.close()

                if task=='WATT3':
                    plot_design_timeseries(subjinfo, 0, 200)
                else:
                    plot_design_timeseries(subjinfo, 0, 100)
                plt.gcf()
                plt.savefig(os.path.join(fig_dir, '%s_%s__design_timeseries_trunc' % (sub, task)))
                plt.close()                

                plot_design_heatmap(subjinfo)
                plt.gcf()
                plt.savefig(os.path.join(fig_dir, '%s_%s_heatmap' % (sub, task)), bbox_inches="tight")
                plt.close()

                plot_vif(subjinfo)
                plt.gcf()
                plt.savefig(os.path.join(fig_dir, '%s_%s__design_vif' % (sub, task)))
                plt.close()
            except:
                print('task not found')




# # First Level Visualization

# In[ ]:

if run_first_level:
    for task in tasks:
        contrast_maps = glob(path.join(first_level_dir, '*', task, '*maps*', '*.nii.gz'))
        for map_file in contrast_maps:
            contrast_name = map_file[map_file.index('contrast')+9:].replace('.nii.gz', '')
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
def transform_p_val_map(map_path):
    img = nib.load(map_path)
    p_vals = img.get_fdata()
    p_vals[p_vals==0.0] = np.nan
    p_vals = 1 - p_vals
    neg_log_pvals = -np.log10(p_vals)
    return nib.Nifti1Image(neg_log_pvals, img.affine, img.header)


def make_mask(img_path):
    return math_img('img > .95',
                    img=nib.load(img_path))


def mask_img(img_path, mask):
    img = nib.load(img_path)
    data = img.get_fdata()
    data[~mask.get_fdata().astype(bool)] = np.nan
    return nib.Nifti1Image(data, img.affine, img.header)

def double_mask_img(img_path, mask1, mask2):
    img = nib.load(img_path)
    data = img.get_fdata()
    data[(~mask1.get_fdata().astype(bool)) & (~mask2.get_fdata().astype(bool))] = np.nan
    return nib.Nifti1Image(data, img.affine, img.header)

def get_rand_idx(path):
    return str(re.findall(r'[0-9$,%]+\d*', path)[-1])

if run_second_level:
    print('running!')
    for task in tasks:
        print(task)
        contrast_dirs = sorted(glob(path.join(second_level_dir, task, '*maps')))
        for contrast_dir in contrast_dirs:
            RT_flag = 'RT-True' in contrast_dir
            contrast_title = task+'_RT-'+str(RT_flag)
            out_dir = contrast_dir if group == 'NONE' else path.join(contrast_dir, group)
            print(out_dir)
            beta_maps = sorted(glob(path.join(out_dir, 'contrast-*.nii.gz')))
            scnd_lvl_contrasts = list(
                {
                    s.split('2ndlevel-')[-1].replace('.nii.gz', '')
                    for s in beta_maps
                }
            )
            # order the randomise output files to match the order of the beta files
            ordered_file_dict = {
                'tstatPos': [],
                'tfce_corrp_tstatPos': [],
                'tstatNeg': [],
                'tfce_corrp_tstatNeg': [],
            }
            for beta_map in beta_maps:
                scnd_lvl = beta_map.split('2ndlevel-')[-1].replace('.nii.gz', '')
                randomise_dir = beta_map.replace('2ndlevel-%s.nii.gz' % scnd_lvl, 'Randomise')

                with open(path.join(randomise_dir, 't_name_map.json'), 'r') as f:
                    t_name_map = json.load(f)
                for key in ordered_file_dict:
                    direction = 'Pos' if 'Pos' in key else 'Neg'
                    curr_files = [f for f in glob(path.join(randomise_dir, 'randomise_%s*' % key.replace(direction, ''))) if all(s in t_name_map[get_rand_idx(f)] for s in [scnd_lvl, direction])]
                    assert len(curr_files)==1
                    ordered_file_dict[key] += curr_files
            
            # plot each of the second level contrasts separately
            for scnd_lvl in scnd_lvl_contrasts:
                curr_title = contrast_title + '_2ndlevel-'+scnd_lvl
                curr_idx = [i for i,f in enumerate(beta_maps) if scnd_lvl in f]
                
                curr_beta_maps = [beta_maps[i] for i in curr_idx]
                curr_tpPos_maps = [ordered_file_dict['tfce_corrp_tstatPos'][i] for i in curr_idx]
                curr_tstatPos_maps = [ordered_file_dict['tstatPos'][i] for i in curr_idx]
                curr_tpNeg_maps = [ordered_file_dict['tfce_corrp_tstatNeg'][i] for i in curr_idx]
                curr_tstatNeg_maps = [ordered_file_dict['tstatNeg'][i] for i in curr_idx]
                curr_masked_beta_images = [double_mask_img(beta_path, make_mask(mask_path1), make_mask(mask_path2)) for beta_path, mask_path1, mask_path2 in zip(curr_beta_maps, curr_tpPos_maps, curr_tpNeg_maps)]
                curr_masked_tstat_images = [double_mask_img(tstat_path, make_mask(mask_path1), make_mask(mask_path2)) for tstat_path, mask_path1, mask_path2 in zip(curr_tstatPos_maps, curr_tpPos_maps, curr_tpNeg_maps)]
                contrast_titles = [get_contrast_title(path) for path in curr_beta_maps]

                # plot and save
                # Betas
                plot_task_maps(curr_beta_maps, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_Beta-raw.pdf'%scnd_lvl))
                plot_task_maps(curr_masked_beta_images, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_Beta-tpvalMasked.pdf' %scnd_lvl))
                # T tests > 0
                plot_task_maps(curr_tpPos_maps, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_TpvalPos-raw.pdf' %scnd_lvl))
                plot_task_maps(curr_tpPos_maps, curr_title, threshold=.95, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_TpvalPos-thresh95.pdf' %scnd_lvl))
                plot_task_maps(curr_tstatPos_maps, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_TstatsPos-raw.pdf' %scnd_lvl))
                plot_task_maps(curr_masked_tstat_images, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_Tstats-tpvalMasked.pdf' %scnd_lvl))
                # T tests < 0
                plot_task_maps(curr_tpNeg_maps, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_TpvalNeg-raw.pdf' %scnd_lvl))
                plot_task_maps(curr_tpNeg_maps, curr_title, threshold=.95, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_TpvalNeg-thresh95.pdf' %scnd_lvl))
                plot_task_maps(curr_tstatNeg_maps, curr_title, threshold=0, contrast_titles=contrast_titles).savefig(path.join(out_dir, 'AAPLOTS_2ndlevel-%s_TstatsNeg-raw.pdf' %scnd_lvl))
