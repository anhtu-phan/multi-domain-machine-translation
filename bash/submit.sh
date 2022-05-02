#!/bin/bash
#
#SBATCH --job-name=nmt_training
#SBATCH --output=nmt_training_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /home/students/anhtu/multi-domain-machine-translation/bash/run_training.sh