for task in ANT CCTHot DPX discountFix motorSelectiveStop stopSignal stroop twoByTwo WATT3
do
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}/g"  2ndlevel_analysis.batch | sbatch 
done

