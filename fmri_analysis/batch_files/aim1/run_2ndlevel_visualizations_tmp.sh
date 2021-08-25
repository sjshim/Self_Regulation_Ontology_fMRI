#aim1
for task in ANT DPX discountFix stroop WATT3 
do
    sed -e "s/{task}/$task/g"  2ndlevel_visualization.batch | sbatch 
done
