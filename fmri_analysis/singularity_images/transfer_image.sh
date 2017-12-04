sshfs sherlock:/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/fmri_analysis /mnt/temp
rm -f /mnt/temp/singularity_images/*img
cp *img /mnt/temp/singularity_images/
sudo umount /mnt/temp