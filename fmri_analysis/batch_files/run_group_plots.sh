for task in stroop ANT
do
    sed "s/{task}/$task/g" group_plots.batch | sbatch -p russpold
done

