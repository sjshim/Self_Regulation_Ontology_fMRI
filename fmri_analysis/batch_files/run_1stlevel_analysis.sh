derivatives_loc=`sed '6q;d' singularity_config.txt`
data_loc=`sed '8q;d' singularity_config.txt`
analysis_flag=""

for task in stopSignal stroop
do
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}/--rt/g"  1stlevel_analysis.batch | sbatch 
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}//g"  1stlevel_analysis.batch | sbatch
done

