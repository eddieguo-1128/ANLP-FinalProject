#!/bin/bash
#SBATCH --job-name=validation
#SBATCH --output=validation.out
#SBATCH --error=validation.err
#SBATCH --partition=babel-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tianyueo@andrew.cmu.edu


source ~/.bashrc
conda activate agent_eval

REC_FILE="/home/tianyueo/agent/fine_tune/mistral_full_validation_2"

python validation.py \
--checkpoints-folder /data/tir/projects/tir2/users/tianyueo/output_5k_full_mistral_flsh_attn_2 \
--record-file-path "$REC_FILE" \
--eval-file /home/tianyueo/agent/data/final_output_full_injected_llama2_val.json \

sort -t '-' -k2,2n "$REC_FILE" -o "$REC_FILE"
