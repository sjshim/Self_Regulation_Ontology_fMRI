from glob import glob
import os
import pandas as pd
from expanalysis.experiments import processing
from behavioral_data_prep.jspsych_processing import (calc_ANT_DV)


def calc_exp_DVs(df, use_check = True, use_group_fun = True, group_kwargs=None):
    '''Function to calculate dependent variables
    :experiment: experiment key used to look up appropriate grouping variables
    :param use_check: bool, if True exclude dataframes that have "False" in a 
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    '''
    lookup = {'attention_network_task': calc_ANT_DV}
    assert (len(df.experiment_exp_id.unique()) == 1), "Dataframe has more than one experiment in it"
    exp_id = df.experiment_exp_id.unique()[0]
    fun = lookup.get(exp_id, None)
    if group_kwargs is None:
        group_kwargs = {}
    if fun:
        try:
            DVs,description = fun(df, use_check=use_check, use_group_fun=use_group_fun, kwargs=group_kwargs)
        except TypeError:
            DVs,description = fun(df, use_check)
        DVs, valence = processing.organize_DVs(DVs)
        return DVs, valence, description
    else:
        return None, None, None
    
def get_exp_DVs():
    file_dir = os.path.dirname(__file__)
    # calculate DVs
    group_kwargs = {'samples': 50,
                    'burn': 10,
                    'thin': 1}
    exp_DVs = {}
    for task_data in glob(os.path.join(file_dir, '../behavioral_data/processed/group_data/*csv')):
        df = pd.read_csv(task_data)
        exp_id = df.experiment_exp_id.unique()[0]
        print(exp_id)
        # Eperiments whose analysis has been overwritten in this repo
        DVs, valence, description = calc_exp_DVs(df, 
                                                 use_group_fun=True,
                                                 group_kwargs=group_kwargs)
        # else use default expanalysis
        if DVs is None:
            DVs, valence, description = processing.calc_exp_DVs(df, 
                                                    use_group_fun=True,
                                                    group_kwargs=group_kwargs)
        exp_DVs[exp_id] = DVs
    DV_df = pd.DataFrame()
    for name, DV in exp_DVs.items():
        DV.columns = [name+'_%s' % i for i in DV.columns]
        DV_df = pd.concat([DV_df, DV], axis=1) 
    return DV_df
