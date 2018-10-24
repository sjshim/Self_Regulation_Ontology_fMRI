import argparse
from glob import glob
import os
import pandas as pd
from expanalysis.experiments import processing

def get_exp_DVs(use_group_fun=True, group_kwargs=None, out_dir=None):
    file_dir = os.path.dirname(__file__)
    # calculate DVs
    if group_kwargs is None:
        group_kwargs = {}
    exp_DVs = {}
    for task_data in glob(os.path.join(file_dir, '../behavioral_data/processed/group_data/*.csv')):
        df = pd.read_csv(task_data)
        exp_id = df.experiment_exp_id.unique()[0]
        print(exp_id)
        if out_dir:
            group_kwargs['outfile'] = os.path.join(out_dir, exp_id)
        DVs, valence, description = processing.calc_exp_DVs(df, 
                                                use_group_fun=use_group_fun,
                                                group_kwargs=group_kwargs)
        exp_DVs[exp_id] = DVs
        if out_dir:
            DVs.to_pickle(os.path.join(out_dir, exp_id+'_DVs.pkl'))
    DV_df = pd.DataFrame()
    for name, DV in exp_DVs.items():
        if DV is not None:
            DV.columns = [name+'_%s' % i for i in DV.columns]
            DV_df = pd.concat([DV_df, DV], axis=1) 
    return DV_df

if __name__ =='__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_group', action='store_false')
    # HDDM params
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--hddm_samples', default=20000, type=int)
    parser.add_argument('--hddm_burn', default=10000, type=int)
    parser.add_argument('--hddm_thin', default=1, type=int)
    parser.add_argument('--no_parallel', action='store_false')
    parser.add_argument('--num_cores', default=None, type=int)
    parser.add_argument('--mode', default=None, type=str)
    
    args = parser.parse_args()
    out_dir = args.out_dir
    use_group = args.no_group
    # HDDM variables
    hddm_samples = args.hddm_samples
    hddm_burn= args.hddm_burn
    hddm_thin= args.hddm_thin
    parallel = args.no_parallel
    num_cores = args.num_cores
    # mode for motor selective stop signal
    mode = args.mode
       
    #calculate DVs
    group_kwargs = {'parallel': parallel,
                    'num_cores': num_cores,
                    'samples': hddm_samples,
                    'burn': hddm_burn,
                    'thin': hddm_thin}
    
    DV_df = get_exp_DVs(use_group, group_kwargs, out_dir)
    if out_dir is not None:
        DV_df.to_pickle(os.path.join(out_dir, 'fmri_DVs.pkl'))
    

