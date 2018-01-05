#!/bin/bash
docker build --rm -t sro_fmri .
`docker rmi $(docker images -f 'dangling=true' -q)` # remove dangling docker images
rm -f singularity_images/*img
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v /home/ian/Experiments/expfactory/Self_Regulation_Ontology_fMRI/singularity_images:/output --privileged -t singularityware/docker2singularity sro_fmri
echo Finished Conversion
cd singularity_images
bash transfer_image.sh
echo Finished Transfer
