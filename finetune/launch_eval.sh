#!/bin/bash
#SBATCH --job-name=s_llama
#SBATCH --output=s_llama.out
#SBATCH --error=s_llama.err
#SBATCH --partition=babel-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate agent

python eval.py --eval_set /home/tianyueo/agent/data/final_output_full_injected_llama2_test.json