for task in ANT CCTHot DPX discountFix motorSelectiveStop stopSignal stroop twoByTwo WATT3
do
    sed -e "s/{task}/$task/g"  timeseries_parcel.batch | sbatch 
done
