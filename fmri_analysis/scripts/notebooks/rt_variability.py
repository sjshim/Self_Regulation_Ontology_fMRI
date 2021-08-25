# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
# ---

# %% [markdown]
#  ## RT modeling 
#  

# %%
import pandas as pd
from pathlib import Path
from scipy.stats import exponnorm
from scipy import stats

# %%


def get_switchdata(datadir):
    switchdata = pd.read_csv(datadir / Path('threebytwo.csv.gz'), index_col=0)

    # remove practice trials
    switchdata = switchdata.query('exp_stage == "test"')

    # remove implausible response times
    switchdata = switchdata.query('rt > 200')

    # remove incorrect trials
    switchdata = switchdata.query('correct == True')

    # convert variables to string so that they will be treated as factors
    for var in ['CTI', 'stim_number']:
        switchdata[var] = [str(i) for i in switchdata[var]]

    # create an overall switch vs. nonswitch variable
    switchdata['switch'] = (switchdata['task_switch'] != 'stay').astype('int')
    assert 'rt' in switchdata.columns
    return(switchdata)



# %%

if __name__ == "__main__":
    datadir = Path('/Users/poldrack/Dropbox/code/psych253/2020/data/SRO')
    switchdata = get_switchdata(datadir)
    max_k = 10
    min_k = 0.2
    ksthresh = 0.05
    # get exgauss fits for each subject for nonswiich trials
    param_fits = []
    for worker_id in switchdata.worker_id.unique():
        worker_data = switchdata.query(f'worker_id == "{worker_id}" & switch == 0')
        subfit = exponnorm.fit(worker_data.rt)
        ks = stats.kstest(worker_data.rt, "exponnorm", subfit)
        param_fits.append(subfit + (worker_data.rt.mean(), ks.pvalue < ksthresh))
    
    param_df = pd.DataFrame(data=param_fits, columns=['k', 'loc', 'scale', 'meanRT', 'poor_fit'])
    param_df_clean = param_df.query(f"k > {min_k} & k < {max_k} & poor_fit == False")

    print('Descriptive stats:')
    print(param_df_clean.describe())

    meanrt_params = exponnorm.fit(param_df_clean.meanRT)
    print(meanrt_params)
    print(stats.kstest(param_df_clean.meanRT, "exponnorm", meanrt_params))

    xvals = np.arange(0, 2000)
    fitted_dist = exponnorm.pdf(xvals, meanrt_params[0], loc=meanrt_params[1], scale=meanrt_params[2])
    fig, ax = plt.subplots()
    ax.hist(param_df_clean.meanRT, density=True, bins=50)
    ax.plot(xvals, fitted_dist, 'k')
    plt.title('ex-Gaussian fit to mean RT distribution')
    plt.savefig('meanrt_fit.pdf')

    # plot all pdfs for individual subjects
    fig, ax = plt.subplots()
    for i in param_df_clean.index:
        sub_params = param_df_clean.loc[i, ['k', 'loc', 'scale']].tolist()
        fitted_dist = exponnorm.pdf(xvals,
                                    sub_params[0], 
                                    loc=sub_params[1],
                                    scale=sub_params[2])

        ax.plot(xvals, fitted_dist, 'k', linewidth=0.1, alpha=0.5)
    plt.title('ex-Gaussian fits to individual RT distribution')
    plt.savefig('individual_rt_fit.pdf')



# %%
