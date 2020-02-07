> preproc_task_list.sh
#bash add_mriqc.sh
bash add_fmriprep.sh
#bash add_freesurfer.sh
lines=$(wc -l < "preproc_task_list.sh")
sed  -i "3s/.*/#SBATCH --array=1-${lines}%12 /" preproc.batch

