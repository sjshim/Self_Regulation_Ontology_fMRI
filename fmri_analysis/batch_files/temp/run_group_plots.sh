sed -e "s/{output}/output/g"  group_plots.batch | sbatch -p russpold
sed -e "s/{output}/output_noRT/g"  group_plots.batch | sbatch -p russpold

