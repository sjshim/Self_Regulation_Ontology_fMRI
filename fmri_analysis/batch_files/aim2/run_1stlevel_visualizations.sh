#aim2
for task in manipulationTask motorSelectiveStop stopSignal discountFix 
do
    sed -e "s/{task}/$task/g"  1stlevel_visualization.batch | sbatch 
done