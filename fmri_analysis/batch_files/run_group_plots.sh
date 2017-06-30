for task in ANT CCTHot DPX motorSelectiveStop stopSignal stroop twoByTwo
do
    sed "s/{task}/$task/g" group_plots.batch | sbatch -p russpold
done

