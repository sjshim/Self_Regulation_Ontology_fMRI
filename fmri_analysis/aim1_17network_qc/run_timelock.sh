for task in ANT CCTHot discountFix DPX motorSelectiveStop stopSignal stroop twoByTwo WATT3
do
    sed  -e "s/{task}/$task/g" network_timelocked.batch | sbatch
done
