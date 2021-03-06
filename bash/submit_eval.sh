#!/bin/bash
#
#SBATCH --job-name=ted_mul
#SBATCH --output=nmt_eval_output_de_en_ted_mutil.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun bash/run_eval.sh