# 2ndlevel intercept (group mean) contrasts
for task in ANT CCTHot DPX discountFix motorSelectiveStop stopSignal stroop twoByTwo WATT3
do
    sed -e "s/{task}/$task/g" -e "s/{scnd_lvl}/intercept/g" 2ndlevel_analysis.batch | sbatch
done