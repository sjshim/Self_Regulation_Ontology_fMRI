
from glob import iglob, glob
import shutil
import os
import numpy as np
import nibabel as nib
import argparse

from nimsphysio.nimsphysio import NIMSPhysio

# # Download Rest data if it hasn't been downloaded already
# import flywheel
# import tarfile

# project_name = 'russpold/uh2'
# fw = flywheel.Client()
# project = fw.lookup(project_name)
# fw.download_tar(project.sessions(), 'gephysio.tar', include_types=['gephysio'])
# tarfile.open('gephysio.tar').extractall()

# # 2 sessions were mislabeled during acquisition
# os.makedirs('scitran/russpold/uh2/s130/')
# shutil.move('scitran/russpold/uh2/s130_2/14598', 'scitran/russpold/uh2/s130')

# os.makedirs('scitran/russpold/uh2/s525/')
# shutil.move('scitran/russpold/uh2/s525_2/14497', 'scitran/russpold/uh2/s525')

def get_args():
    parser = argparse.ArgumentParser(description='aim1 rest physio regression')
    parser.add_argument('-fw_dir', default='/home/groups/russpold/uh2_analysis/Self_Regulation_Ontology_fMRI/fmri_data_prep/scitran/russpold/uh2/')
    parser.add_argument('-fmriprep_dir', default='/oak/stanford/groups/russpold/data/uh2/aim1/BIDS_scans/derivatives/fmriprep')
    parser.add_argument('-firstlevel_dir', default='/oak/stanford/groups/russpold/data/uh2/aim1/BIDS_scans/derivatives/1stlevel')
    parser.add_argument('--slice_window', default=.085, help='time in seconds for acquiring a single slice')
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()

    sub_sessions_missing_physio = []
    for rest_nii in iglob(
        os.path.join(
            args.fmriprep_dir,
            'sub-s*/ses-*/func/*_task-rest_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            )
        ):
        sub = rest_nii.split('sub-')[-2].split('/')[0]
        ses = rest_nii.split('ses-')[-2].split('/')[0]
        run = rest_nii.split('run-')[-1].split('_')[0]
        
        physio_name = sub + f'_{ses}' if int(ses) > 1 else sub

        physio_files = sorted(
            glob(
                os.path.join(
                    args.fw_dir,
                    f'{physio_name}/*/task_rest_run_{run}*/*gephysio.zip'
                    )
                )
            )
        if len(physio_files)==0:
            sub_sessions_missing_physio.append(f'sub-{sub}_ses-{ses}_run-{run}')
        else:
            outdir = os.path.join(args.firstlevel_dir, sub, 'rest', f'ses-{ses}')
            os.makedirs(outdir, exist_ok=True)
            outnii_name = rest_nii.split('/')[-1].replace('preproc_bold', 'preprocPhsyio_bold')

            niimg = nib.load(rest_nii)
            phys = NIMSPhysio(
                physio_files[-1], # I confirmed that for subjects with 2, the last was used
                tr=niimg.header.get_zooms()[3],
                nframes=niimg.shape[3],
                slice_onsets=[0] * niimg.shape[2], #slice time corrected, all "onset" at 0
                slice_window=float(args.slice_window)
            )
            # save regressor info
            np.savetxt(os.path.join(outdir, 'resp.txt'), phys.resp_wave)
            np.savetxt(os.path.join(outdir, 'pulse.txt'), phys.card_trig)
            np.savetxt(os.path.join(outdir, 'slice_onsets.txt'), phys.slice_onsets)
            phys.write_regressors(os.path.join(outdir, 'regs.txt'))

            # regress out physio, save result
            d_corrected, PCT_VAR_REDUCED = phys.denoise_image(niimg.get_fdata(), phys.regressors)
            np.save(os.path.join(outdir, 'pct_var_reduced.npy'), PCT_VAR_REDUCED)
            nib.save(
                nib.Nifti1Image(d_corrected, niimg.affine, niimg.header),
                os.path.join(outdir, outnii_name)
                )
    print('the following sessions did not appear to have rest physio:')
    print('\n'.join(sub_sessions_missing_physio))
