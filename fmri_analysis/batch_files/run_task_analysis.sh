for path in /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-s1??
do
    sid=${path: -4}
    sed "s/{sid}/$sid/g" task_analysis.batch | sbatch -p russpold
done