# 2ndlevel RT contrasts, for Jeanette
for task in ANT discountFix stroop twoByTwo
do
    sed -e "s/{task}/$task/g" -e "s/{scnd_lvl}/task_RT/g" 2ndlevel_analysis.batch | sbatch
done

sed -e "s/{task}/DPX/g" -e "s/{scnd_lvl}/AX_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/DPX/g" -e "s/{scnd_lvl}/AY_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/DPX/g" -e "s/{scnd_lvl}/BX_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/DPX/g" -e "s/{scnd_lvl}/BY_RT/g" 2ndlevel_analysis.batch | sbatch

sed -e "s/{task}/CCTHot/g" -e "s/{scnd_lvl}/first_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/CCTHot/g" -e "s/{scnd_lvl}/subsequent_RT/g" 2ndlevel_analysis.batch | sbatch

sed -e "s/{task}/motorSelectiveStop/g" -e "s/{scnd_lvl}/crit_go_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/motorSelectiveStop/g" -e "s/{scnd_lvl}/crit_stop_failure_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/motorSelectiveStop/g" -e "s/{scnd_lvl}/noncrit_nosignal_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/motorSelectiveStop/g" -e "s/{scnd_lvl}/noncrit_signal_RT/g" 2ndlevel_analysis.batch | sbatch

sed -e "s/{task}/stopSignal/g" -e "s/{scnd_lvl}/go_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/stopSignal/g" -e "s/{scnd_lvl}/stop_failure_RT/g" 2ndlevel_analysis.batch | sbatch

sed -e "s/{task}/WATT3/g" -e "s/{scnd_lvl}/planning_RT/g" 2ndlevel_analysis.batch | sbatch
sed -e "s/{task}/WATT3/g" -e "s/{scnd_lvl}/acting_RT/g" 2ndlevel_analysis.batch | sbatch
