from glob import glob
from utils.display_utils import get_design_df, plot_design

task = 'stroop'
# location to folders containing design files
files = glob( '/mnt/Sherlock_Scratch/1stLevel/*task_%s' % task)

df = get_design_df(files[0])
plot_design(df)


get_design_df(files[3]).corr().loc['congruent','incongruent']
