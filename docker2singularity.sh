#!/bin/bash
# docker build --rm -t fmri_env .
rm -f singularity_images/*img
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v /Users/henrymj/Documents/uh2/Self_Regulation_Ontology_fMRI/singularity_images:/output --privileged -t singularityware/docker2singularity fmri_env
echo Finished Conversion
cd singularity_images
bash transfer_image.sh
echo Finished Transfer
