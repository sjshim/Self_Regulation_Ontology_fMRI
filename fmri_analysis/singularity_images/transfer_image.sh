sshfs sherlock:/scratch/PI/russpold/work/ieisenbe/uh2 /mnt/Sherlock_Scratch
rm -f /mnt/Sherlock_Scratch/singularity_images/*img
cp *img /mnt/Sherlock_Scratch/singularity_images/
sudo umount /mnt/Sherlock_Scratch