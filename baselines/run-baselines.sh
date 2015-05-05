#!/bin/bash

# Job name
#$ -N ers-baselines

# Use cwd for submitting and writing output
#$ -cwd

# Parallelization settings
#$ -pe shared 1

# Memory per slot
#$ -l mf=8G

# Send email to my gmu account at start and finish
#$ -M msweene2@gmu.edu
#$ -m be

# Load necessary modules
source /etc/profile
module load gcc/4.8.4
module load intel-itac/8.1.4/045

# Start the job
/home/msweene2/anaconda/bin/python \
baselines.py

