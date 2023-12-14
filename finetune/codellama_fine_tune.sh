#!/bin/bash
#SBATCH --job-name=codellama_ql
#SBATCH --output=codellama_ql.out
#SBATCH --error=codellama_ql.err
#SBATCH --partition=babel-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiaruil5@andrew.cmu.edu

source ~/.bashrc
conda activate agent

WANDB_PROJECT=llama torchrun --nproc_per_node=4 --master_port=20002 ../src/fschat/fastchat/train/train_lora.py \
    --model_name_or_path /data/user_data/jiaruil5/.cache/models--codellama--CodeLlama-7b-hf/snapshots/bc5283229e2fe411552f55c71657e97edf79066c \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ../data/final_output_full_injected_llama2_train.json \
    --eval_data_path ../data/final_output_full_injected_llama2_val.json \
    --bf16 True \
    --output_dir /data/user_data/jiaruil5/agent/output_5k_qlora_codellama \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora True \
    --flash_attn True \
    --report_to wandb

