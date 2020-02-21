sed -e "s/{task}/discountFix/g" -e "s/{rt_flag}/--rt/g"  1stlevel_analysis.batch | sbatch 

sed -e "s/{task}/motorSelectiveStop/g" -e "s/{rt_flag}/--rt/g"  1stlevel_analysis.batch | sbatch 
sed -e "s/{task}/motorSelectiveStop/g" -e "s/{rt_flag}//g"  1stlevel_analysis.batch | sbatch
