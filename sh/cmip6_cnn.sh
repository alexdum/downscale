#!/bin/bash 

#This shell is to subset cmip6_raw_il to Illinois domain

#SBATCH --time=168:00:00 ##Set time needed for the job
#SBATCH --job-name=cnn
#SBATCH --output=/data/keeling/a/ad87/downscale/sh/logs/%x_%j_cmip6_raw_il.out  ## Capture standard output to a log file
#SBATCH --error=/data/keeling/a/ad87/dowsncale/sh/logs/%x_%j_cmip6_raw_il.err   ## Capture standard error to a separate log file
#SBATCH --mail-type=BEGIN ##Specify the type of job execution emails you need like beginning, failing or end of job.
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=ad87@illinois.edu ##Specify your email where the information about the execution will be sent.




# Command to request interactive GPU session
srun -p l40s -n 20 -N 1 --mem=150g --gres=gpu:L40S:1 --pty bash

# Activate the conda environment
source ~/miniconda3/bin/activate ml  ## This is the python virtual environment




echo "#####################################################" ##This is to print any info about the job
echo "# Welcome to the CNN subset script"
echo "#####################################################"

###export OMP_NUM_THREADS=1 ##This is to specify how many threads you need per processor.

 
cd /data/keeling/a/ad87/dowscale 

python python/cnn.ipynb ##Execute the program

echo "DONE!"
