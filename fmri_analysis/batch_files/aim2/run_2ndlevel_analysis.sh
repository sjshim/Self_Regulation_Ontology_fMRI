for task in discountFix motorSelectiveStop stopSignal manipulationTask
do
    sed -e "s/{task}/$task/g" 2ndlevel_analysis.batch | sbatch 
done

