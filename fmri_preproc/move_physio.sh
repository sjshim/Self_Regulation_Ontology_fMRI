for path in /oak/stanford/groups/russpold/data/uh2/sub*
do
    sid=${path:(-4)}
    for ses in 1 2
    do
        cp /oak/stanford/groups/russpold/data/sub-${sid}/ses-$ses/func/*recording* /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-$ses/func/
    done
done