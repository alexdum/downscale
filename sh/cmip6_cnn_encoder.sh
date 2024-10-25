#!/bin/bash 

#This shell is to subset cmip6_raw_il to Illinois domain

#SBATCH --partition=l40s        ##Specify the partition
#SBATCH --nodes=1              ##Number of nodes
#SBATCH --ntasks=20            ##Number of CPU cores
#SBATCH --mem=150G             ##Memory per node
#SBATCH --gres=gpu:L40S:1      ##GPU requirement
#SBATCH --time=2-00:00:00       ##Set time needed for the job
#SBATCH --job-name=cnn
#SBATCH --output=/data/keeling/a/ad87/downscale/sh/logs/%x_%j_cmip6_raw_il.out
#SBATCH --error=/data/keeling/a/ad87/downscale/sh/logs/%x_%j_cmip6_raw_il.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=ad87@illinois.edu



# Activate the conda environment
source ~/miniconda3/bin/activate ml  ## This is the python virtual environment
module load GPU



echo "#####################################################" ##This is to print any info about the job
echo "# Welcome to the CNN subset script"
echo "#####################################################"


 
cd /data/keeling/a/ad87/downscale

python python/cnn_encoder_decoder.py ##Execute the program

echo "DONE!"
