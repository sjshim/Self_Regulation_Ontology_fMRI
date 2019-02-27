from glob import glob
from nilearn import image

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
    rt_flag = "True" if regress_rt else "False"
    beta_flag = "True" if beta else "False"
    maps_dir = path.join(second_level_dir, task, 'secondlevel_RT-%s_beta-%s_maps' % (rt_flag, beta_flag))
    map_files = glob(path.join(maps_dir, '*'))
    maps = {}
    for f in map_files:
        name = f.split('-')[-1]
        maps[name] = image.load_img(f)
    return maps