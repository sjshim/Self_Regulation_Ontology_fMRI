sshfs sherlock:/scratch/PI/russpold/work/ieisenbe/uh2 /mnt/temp
rm -f /mnt/temp/singularity_images/*img
cp *img /mnt/temp/singularity_images/
sudo umount /mnt/temp