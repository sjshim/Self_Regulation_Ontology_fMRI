from glob import glob
from nilearn import image
from os import path
from utils.utils import get_flags

def create_group_mask(fmriprep_dir, threshold=.8, verbose=True):
    if verbose:
        print('Creating Group mask...')
    brainmasks = glob(path.join(fmriprep_dir,'sub-s???',
                               '*','func','*MNI152NLin2009cAsym_brainmask*'))
    mean_mask = image.mean_img(brainmasks)
    group_mask = image.math_img("a>=%s" % str(threshold), a=mean_mask)
    return group_mask
    if verbose:
        print('Finished creating group mask')



def load_contrast_maps(second_level_dir, task, regress_rt=False, beta=False):
    rt_flag, beta_flag = get_flags(regress_rt, beta)
    maps_dir = path.join(second_level_dir, task, 'secondlevel_RT-%s_beta-%s_N-*_maps' % (rt_flag, beta_flag))
    maps_dirs = glob(maps_dir)
    if len(maps_dirs) > 1:
        maps_dir = sorted(maps_dirs, key=lambda x: x.split('_')[-2])[-1]
    else:
        maps_dir = maps_dirs[0]
    map_files = glob(path.join(maps_dir, '*'))
    maps = {}
    for f in map_files:
        name = f.split(path.sep)[-1][9:].rstrip('.nii.gz')
        maps[name] = image.load_img(f)
    return maps