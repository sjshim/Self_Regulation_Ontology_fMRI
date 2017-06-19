for path in /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-s???
do
    sid=${path: -4}
    if [ -f /scratch/PI/russpold/work/ieisenbe/uh2/output/datasink/1stLevel/${sid}_task_DPX/cope1.nii.gz ]; then
        echo task analysis already run on $sid DPX, skipping all tasks
    else
    	sed "s/{sid}/$sid/g" task_analysis.batch | sbatch -p russpold
    fi
done

