# aim1_derivatives_loc=`sed '6q;d' singularity_config.txt`
# aim2_derivatives_loc=`sed '6q;d' ../aim2/singularity_config.txt`

#aim1
for task in manipulationTask motorSelectiveStop stopSignal discountFix 
do
    sed -e "s/{task}/$task/g"  1stlevel_visualization.batch | sbatch 
done

# #aim2
# for task in manipulationTask motorSelectiveStop stopSignal discountFix
#     sed -e "s/{task}/$task/g" -e "s/{derivatives_loc}/$aim2_derivatives_loc/g"  1stlevel_visualization.batch | sbatch 
# done