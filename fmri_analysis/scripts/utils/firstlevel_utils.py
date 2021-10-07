from collections import namedtuple
from glob import glob
from nistats.design_matrix import make_first_level_design_matrix
import numpy as np
import os
import json
from os import makedirs, path
import pandas as pd
import patsy
import pickle
import warnings
from utils.events_utils import get_beta_series, parse_EVs
from utils.utils import get_contrasts, get_flags

# ********************************************************
# helper functions
# ********************************************************


def temp_deriv(dataframe, columns=None):
    if columns is None:
        columns = dataframe.columns
    td = dataframe.loc[:, columns].apply(np.gradient)
    td.iloc[0, :] = 0
    for i, col in td.iteritems():
        insert_loc = dataframe.columns.get_loc(i)
        dataframe.insert(insert_loc+1, i+'_TD', col)


def save_new_EVs(events, subjinfo, output_dir, beta=True, regress_rt=False):

    # get filepath/name
    subj, task = subjinfo.ID.split('_')
    directory = path.join(output_dir, subj, task)
    flags = subjinfo.get_flags()
    filename = path.join(directory, 'simplified_events_%s.csv' % flags)
    makedirs(directory, exist_ok=True)

    if beta:
        EV_dict = get_beta_series(events, regress_rt=regress_rt)
    else:
        EV_dict = parse_EVs(events, task, regress_rt=regress_rt)

    event_df = pd.DataFrame.from_dict(EV_dict)  # change to dataframe

    s = event_df.iloc[0, :]  # convert first regressor into data frame
    out_event_df = pd.DataFrame.from_dict(dict(zip(s.index, s.values)))

    # append remaining regressor df's to first regressor's df
    for ridx in range(1, len(event_df)):
        s = event_df.iloc[ridx, :]
        tmp_df = pd.DataFrame.from_dict(dict(zip(s.index, s.values)))
        out_event_df = pd.concat([out_event_df, tmp_df])

    out_event_df = out_event_df.sort_values(by=['onsets', 'conditions'])
    out_event_df.to_csv(filename, index=False)


def create_design(events, confounds, task, TR, subject_id,
                  beta=True, regress_rt=False):
    """
    takes event file and confounds, and creates EV_dict, which is passed
    to make_first_level_design to create a the design matrix.
    """
    if beta:
        EV_dict = get_beta_series(events, regress_rt=regress_rt)
    else:
        EV_dict, meta_dict = parse_EVs(events, task,
                                       regress_rt=regress_rt,
                                       return_metadict=True)

    paradigm = get_paradigm(EV_dict)
    # make design
    n_scans = int(confounds.shape[0])
    design = make_first_level_design_matrix(
        np.arange(n_scans)*TR+TR/2,
        paradigm,
        hrf_model='spm',
        #period_cut=80,
        drift_model='cosine',
        add_regs=confounds.values,
        add_reg_names=list(confounds.columns))
    # add temporal derivative to task columns
    task_cols = [i for i in paradigm.trial_type.unique() if i != 'junk']
    temp_deriv(design, task_cols)
    return (design, meta_dict)


def add_FD_meta(meta_dict, subject_id, task, mriqc_dir):
    json_files = glob(path.join(mriqc_dir,
                                '*%s*' % subject_id,
                                'ses-*/func',
                                '*%s*_bold.json' % task))
    assert len(json_files) <= 1
    if len(json_files) != 0:
        with open(json_files[0], 'r') as f:
            mriqc_dict = json.load(f)
            meta_dict['FD_mean'] = mriqc_dict.get('fd_mean')
            meta_dict['FD_num'] = mriqc_dict.get('fd_num')
            meta_dict['FD_perc'] = mriqc_dict.get('fd_perc')
    return meta_dict


def make_first_level_obj(subject_id, task,
                         fmriprep_dir, data_dir, output_dir,
                         TR,
                         regress_rt=False,
                         beta=False,
                         a_comp_cor=True,
                         use_aroma=False):
    """
    retrieves and passes func_file, mask_file, events, confounds, design,
    and contrasts to FirstLevel class and returns subjinfo object,
    prints error if no func or mask file
    """
    func_file, mask_file = get_func_file(fmriprep_dir, subject_id, task,
                                         use_aroma=use_aroma)
    if func_file is None or mask_file is None:
        print("Missing MRI files for %s: %s" % (subject_id, task))
        return None
    events = get_events(data_dir, subject_id, task)
    if events is None:
        print("Missing event files for %s: %s" % (subject_id, task))
        return None
    confounds = get_confounds(fmriprep_dir, subject_id, task,
                              a_comp_cor=a_comp_cor,
                              use_aroma=use_aroma)
    contrasts = get_contrasts(task, regress_rt)
    design, meta_des_dict = create_design(events, confounds, task, TR, subject_id,
                                          beta=beta, regress_rt=regress_rt)
    # add on FD info to meta_dict
    assert fmriprep_dir[-1] != '/'
    deriv_base = path.dirname(fmriprep_dir)
    assert 'fmriprep' not in deriv_base
    meta_dict = add_FD_meta(meta_des_dict,
                            subject_id,
                            task,
                            path.join(deriv_base, 'mriqc'))

    # make subjinfo object
    subjinfo = FirstLevel(func_file, mask_file, events, design, contrasts,
                          '%s_%s' % (subject_id, task),
                          meta_dict)
    subjinfo.model_settings['beta'] = beta
    subjinfo.model_settings['regress_rt'] = regress_rt

    events = get_events(data_dir, subject_id, task)
    save_new_EVs(events, subjinfo, output_dir,
                 beta=beta, regress_rt=regress_rt)
    print(subjinfo)
    return subjinfo


def save_first_level_obj(subjinfo, output_dir, save_maps=False):
    """
    Gets or Creates a directory for saving the first level analyses,
    will also save contrast maps if flagged to do so.
    """
    subj, task = subjinfo.ID.split('_')
    directory = path.join(output_dir, subj, task)
    flags = subjinfo.get_flags()
    filename = path.join(directory, 'firstlevel_%s.pkl' % flags)
    makedirs(directory, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(subjinfo, f)
    if save_maps:
        maps_dir = path.join(directory, 'maps_%s' % flags)
        makedirs(maps_dir, exist_ok=True)
        for name, contrast in subjinfo.contrasts:
            try:
                contrast_map = subjinfo.fit_model.compute_contrast(contrast)
                contrast_file = path.join(maps_dir,
                                          'contrast-%s.nii.gz' % name)
                contrast_map.to_filename(contrast_file)
            except patsy.PatsyError:
                warnings.warn('Contrast: %s failed for %s, %s' %
                              (name, subj, task))


def get_first_level_objs(subject_id, task, first_level_dir,
                         regress_rt=False, beta=False):
    """ gets and returns filepath to first level objects if they exist"""

    rt_flag, beta_flag = get_flags(regress_rt, beta)
    files = path.join(first_level_dir, subject_id, task,
                      'firstlevel*%s_%s*pkl' % (rt_flag, beta_flag))
    return glob(files)


def load_first_level_objs(task, first_level_dir,
                          regress_rt=False, beta=False):
    subjinfos = []
    files = get_first_level_objs('*', task, first_level_dir,
                                 regress_rt=regress_rt, beta=beta)
    for filey in files:
        with open(filey, 'rb') as f:
            subjinfos.append(pickle.load(f))
    return subjinfos


def get_first_level_maps(subject_id, task, first_level_dir, contrast,
                         regress_rt=False, beta=False):
    rt_flag, beta_flag = get_flags(regress_rt, beta)
    files = path.join(
        first_level_dir, subject_id, task,
        'maps_%s_%s/contrast-%s.nii.gz' % (rt_flag, beta_flag, contrast))
    return sorted(glob(files))

def get_first_level_metas(subject_id, task, first_level_dir,
                         regress_rt=False, beta=False):
    rt_flag, beta_flag = get_flags(regress_rt, beta)
    files = path.join(
        first_level_dir, subject_id, task,
        '2ndlevel_meta_%s_%s.json' % (rt_flag, beta_flag))
    return sorted(glob(files))

# ********************************************************
# helper classes
# ********************************************************


SubjInfo = namedtuple('subjinfo',
                      ['func', 'mask', 'design', 'contrasts', 'ID'])


class FirstLevel():
    def __init__(self, func, mask, events, design, contrasts, ID, meta):
        self.func = func
        self.mask = mask
        self.events = events
        self.design = design
        self.contrasts = contrasts
        self.ID = ID
        # for model
        self.model_settings = {'beta': False, 'regress_rt': False}
        self.fit_model = None
        self.meta = meta

    def get_subjinfo(self):
        return SubjInfo(self.func,
                        self.mask,
                        self.design,
                        self.contrasts,
                        self.ID)

    def export_design(self, directory):
        directory = self._get_export_dir(directory)
        flags = self.get_flags()
        self.design.to_csv(path.join(directory, 'design_%s.csv' % flags))

    def export_events(self, directory):
        directory = self._get_export_dir(directory)
        flags = self.get_flags()
        self.events.to_csv(path.join(directory, 'events_%s.csv' % flags))

    def export_2ndlvl_meta(self, directory):
        directory = self._get_export_dir(directory)
        flags = self.get_flags()
        with open(path.join(directory, '2ndlevel_meta_%s.json' % flags), 'w') as f:
            json.dump(self.meta, f)

    def get_flags(self):
        rt_flag, beta_flag = get_flags(self.model_settings['regress_rt'],
                                       self.model_settings['beta'])
        return '%s_%s' % (rt_flag, beta_flag)

    def _get_export_dir(self, directory):
        subj, task = self.ID.split('_')
        directory = path.join(directory, subj, task)
        makedirs(directory, exist_ok=True)
        return directory

    def __str__(self):
        return """
            ** %s **
                * Func File: %s
                * Mask File: %s
                * Model Settings: %s
            """ % (self.ID, self.func, self.mask, self.model_settings)

# ********************************************************
# Process Functions
# ********************************************************


# TODO - change np.gradient (uses forwards and backwards changes) to np.diff (uses only backwards)
def process_confounds(confounds_file, a_comp_cor=True, use_aroma=False):
    """
    scrubbing for TASK
    remove TRs where FD>.5, stdDVARS (that relates to DVARS>.5)
    regressors to use
    ['X','Y','Z','RotX','RotY','RotY','<-firsttemporalderivative','stdDVARs','FD']
    junk regressor: errors, ommissions, maybe very fast RTs (less than 50 ms)
    """
    confounds_df = pd.read_csv(confounds_file, sep='\t',
                               na_values=['n/a']).fillna(0)
    # excessive movement regressors
    excessive_movement = (confounds_df.framewise_displacement > .5) | \
                         (confounds_df.std_dvars > 1.2)
    excessive_movement_TRs = excessive_movement[excessive_movement].index
    excessive_movement_regressors = np.zeros([confounds_df.shape[0],
                                             np.sum(excessive_movement)])
    for i, TR in enumerate(excessive_movement_TRs):
        excessive_movement_regressors[TR, i] = 1
    excessive_movement_regressor_names = ['rejectTR_%d' % TR for TR in
                                          excessive_movement_TRs]

    # base regressors - csf/white matter if aroma, trans/rot otherwise
    if use_aroma:
        base_regressor_df = confounds_df.filter(regex='csf|white_matter').copy()
        del base_regressor_df['csf_wm']
        base_regressor_names = list(base_regressor_df.columns)
        base_regressors = base_regressor_df.values
    else:
        base_regressor_df = confounds_df.filter(regex='trans|rot').copy()
        base_regressor_names = list(base_regressor_df.columns)
        base_regressors = base_regressor_df.values

    # add additional relevant regressors
    add_regressor_names = []  # ['framewise_displacement']
    if a_comp_cor:
        add_regressor_names += [i for i in confounds_df.columns if
                                'a_comp_cor' in i][:8]
    additional_regressors = confounds_df.loc[:, add_regressor_names].values
    regressors = np.hstack((base_regressors,
                            additional_regressors,
                            excessive_movement_regressors))
    # concatenate regressor names
    regressor_names = base_regressor_names +\
        add_regressor_names +\
        excessive_movement_regressor_names
    return regressors, regressor_names


def process_physio(cardiac_file, resp_file):
    # TODO: add physio processing for rest scans
    pass

# ********************************************************
# Getter functions
# ********************************************************


def get_func_file(fmriprep_dir, subject_id, task, use_aroma=False):
    """
    gets the preproc func and mask files files from fmriprep dir, and returns
    """
    # strip "sub" from beginning of subject_id if provided
    subject_id = subject_id.replace('sub-', '')

    if use_aroma:
        func_regex = '*%s*MNI*smoothAROMAnonaggr_bold.nii.gz' % task
    else:
        func_regex = '*%s*MNI*preproc_bold.nii.gz' % task

    # check if there's a session folder
    if os.path.exists(path.join(fmriprep_dir,
                                'sub-%s' % subject_id,
                                'func')):
        func_file = glob(path.join(fmriprep_dir,
                                   'sub-%s' % subject_id,
                                   'func',
                                   func_regex))
        mask_file = glob(path.join(fmriprep_dir,
                                   'sub-%s' % subject_id,
                                   'func',
                                   '*%s_*MNI*brain_mask.nii.gz' % task))
    else:
        func_file = glob(path.join(fmriprep_dir,
                                   'sub-%s' % subject_id,
                                   '*',
                                   'func',
                                   func_regex))
        mask_file = glob(path.join(fmriprep_dir,
                                   'sub-%s' % subject_id,
                                   '*', 'func',
                                   '*%s_*MNI*brain_mask.nii.gz' % task))
    if func_file:
        print(func_file)

    if not func_file or not mask_file:
        return None, None
    return func_file[0], mask_file[0]


def get_confounds(fmriprep_dir, subject_id, task, **process_kwargs):
    # strip "sub" from beginning of subject_id if provided
    subject_id = subject_id.replace('sub-', '')

    # Get the Confounds File (output of fmriprep)
    # Read the TSV file and convert to pandas dataframe

    # check if there's a session folder
    if os.path.exists(path.join(fmriprep_dir,
                                'sub-%s' % subject_id,
                                'func')):
        # gets confounds_file
        confounds_file = glob(path.join(fmriprep_dir,
                                        'sub-%s' % subject_id,
                                        'func',
                                        '*%s*confounds_timeseries.tsv' % task
                                        )
                              )[0]
    else:
        confounds_file = glob(path.join(fmriprep_dir,
                                        'sub-%s' % subject_id,
                                        '*',
                                        'func',
                                        '*%s*confounds_timeseries.tsv' % task
                                        )
                              )[0]

    regressors, regressor_names = process_confounds(confounds_file,
                                                    **process_kwargs)
    confounds = pd.DataFrame(regressors, columns=regressor_names)
    return confounds


def get_events(data_dir, subject_id, task):
    # Get the Events File if it exists
    # Read the TSV file and convert to pandas dataframe

    # check if there's a ses-* folder
    try:
        if os.path.exists(path.join(data_dir,
                                    'sub-%s' % subject_id,
                                    'func')):

            # returns event_file
            event_file = glob(path.join(data_dir,
                                        'sub-%s' % subject_id,
                                        'func',
                                        'sub-*%s*events.tsv' % task))[0]
        else:
            event_file = glob(path.join(data_dir,
                                        'sub-%s' % subject_id,
                                        '*',
                                        'func',
                                        'sub-*%s*events.tsv' % task))[0]
        events_df = pd.read_csv(event_file, sep='\t')
        return events_df
    except IndexError:
        return None


def get_paradigm(EV_dict):
    # convert nipype format to nistats paradigm
    conditions = []
    onsets = []
    amplitudes = []
    durations = []
    for i in range(len(EV_dict['conditions'])):
        onset = EV_dict['onsets'][i]
        onsets += onset
        # add on conditions
        conditions += [EV_dict['conditions'][i]]*len(onset)
        duration = EV_dict['durations'][i]
        # add on duration, and extend if the array has a length of one
        if len(duration) == 1 and len(onset) > 1:
            duration *= len(onset)
        durations += duration
        # add on amplitude, and extend if the array has a length of one
        amplitude = EV_dict['amplitudes'][i]
        if len(amplitude) == 1 and len(onset) > 1:
            amplitude *= len(onset)
        amplitudes += amplitude
    paradigm = {'trial_type': conditions,
                'onset': onsets,
                'modulation': amplitudes,
                'duration': durations}
    paradigm = pd.DataFrame(paradigm).sort_values(
        by='onset').reset_index(drop=True)
    return paradigm
