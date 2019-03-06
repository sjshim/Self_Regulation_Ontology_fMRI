import matplotlib.pyplot as plt
from scipy.stats import norm
from nilearn import image, plotting
from nistats.reporting import plot_design_matrix, plot_contrast_matrix

def plot_design(subjinfo, plot_contrasts=False):
    fig, ax = plt.subplots(figsize=(15,8))
    plot_design_matrix(subjinfo.design, ax=ax, rescale=True)
    if plot_contrasts:
        for name, contrast in subjinfo.contrasts:
            ax=plot_contrast_matrix(contrast, design_matrix=subjinfo.design)
            ax.set_xlabel(name)

def plot_contrast(subjinfo, contrast, **kwargs):
    if type(contrast) == int:
        contrast = subjinfo.contrasts[contrast]
        contrast_title = contrast[0]
        z_map = subjinfo.fit_model.compute_contrast(contrast[1])
    else:
        contrast_title = contrast
        z_map = subjinfo.fit_model.compute_contrast(contrast)
    default_args = {'threshold': norm.isf(0.001), 
                    'display_mode': 'ortho'}
    default_args.update(**kwargs)
    plotting.plot_glass_brain(z_map, colorbar=True, 
                          title=contrast_title,
                          plot_abs=False, **default_args)
    
def plot_average_maps(subjects, contrast_keys=None, **kwargs):
    if contrast_keys is None:
        map_keys = subjects[0].maps.keys()
    else:
        map_keys = contrast_keys
    averages = {}
    for key in map_keys:
        try:
            maps = [i.maps[key] for i in subjects]
        except KeyError:
            maps = [i.fit_model.compute_contrast(key) for i in subjects]
        averages[key] = image.mean_img(maps)
    # plot
    for name, average in averages.items():
        default_args = {'threshold': norm.isf(0.001), 
                        'display_mode': 'ortho'}
        default_args.update(**kwargs)
        plotting.plot_glass_brain(average, colorbar=True, 
                              title=name,
                              plot_abs=False, **default_args)