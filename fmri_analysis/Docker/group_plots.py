from glob import glob
from os import path
from utils.display_utils import get_design_df, group_plot, plot_design

data_dir = '/mnt/Sherlock_Scratch/datasink/1stLevel/'
# plot individual subject's contrasts and then the group
group_plot(data_dir, 'stroop', contrast_index=4, plot_individual=True)
# plot all group contrasts'
group_plot(data_dir, 'ANT')


design = get_design_df(glob(path.join(data_dir,'*ANT*'))[0])
plot_design(design)
