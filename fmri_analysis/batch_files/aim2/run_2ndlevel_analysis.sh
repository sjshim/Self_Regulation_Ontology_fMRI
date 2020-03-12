for task in discountFix motorSelectiveStop stopSignal manipulationTask
do
    for group in NONE BED smoking
    do
        sed -e "s/{task}/$task/g" -e "s/{group}/$group/g" 2ndlevel_analysis.batch | sbatch 
    done
done


