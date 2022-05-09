#!/bin/bash
#
#SBATCH --job-name=nmt_eval
#SBATCH --output=nmt_eval_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /home/students/anhtu/multi-domain-machine-translation/bash/run_eval.sh