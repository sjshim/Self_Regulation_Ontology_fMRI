#!/bin/bash
rm -f singularity_images/*img
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/ian/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/singularity_images:/output --privileged -t --rm singularityware/docker2singularity nipype_image
echo Finished Conversion
cd singularity_images
bash transfer_image.sh
echo Finished Transfer
