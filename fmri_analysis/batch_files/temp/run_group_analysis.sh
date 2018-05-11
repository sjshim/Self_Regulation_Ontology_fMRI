sed -e "s/{output}/output/g"  group_analysis.batch | sbatch -p russpold
sed -e "s/{output}/output_noRT/g"  group_analysis.batch | sbatch -p russpold

