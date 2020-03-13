#aim1
for task in ANT CCTHot DPX discountFix motorSelectiveStop stopSignal stroop twoByTwo WATT3 
do
    sed -e "s/{task}/$task/g"  2ndlevel_visualization.batch | sbatch 
done
