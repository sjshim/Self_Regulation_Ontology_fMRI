for task in manipulationTask motorSelectiveStop stopSignal discountFix 
do
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}/--rt/g"  1stlevel_analysis.batch | sbatch 
    sed -e "s/{task}/$task/g" -e "s/{rt_flag}//g"  1stlevel_analysis.batch | sbatch
done

