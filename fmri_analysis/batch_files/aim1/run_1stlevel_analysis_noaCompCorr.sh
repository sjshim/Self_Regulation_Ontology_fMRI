for task in ANT CCTHot DPX discountFix motorSelectiveStop stopSignal stroop twoByTwo WATT3
do
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}/--rt/g"  1stlevel_analysis_noaCompCorr.batch | sbatch 
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}//g"  1stlevel_analysis_noaCompCorr.batch | sbatch
done

