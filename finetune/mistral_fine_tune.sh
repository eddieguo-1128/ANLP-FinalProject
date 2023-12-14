#!/bin/bash
#SBATCH --job-name=mistral_ql
#SBATCH --output=mistral_ql.out
#SBATCH --error=mistral_ql.err
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


# torchrun --nproc_per_node=4 --master_port=20001 ~/FastChat/fastchat/train/train_mem.py \
#     --model_name_or_path /data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2.json \
#     --bf16 True \
#     --output_dir output_5k \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

WANDB_PROJECT=llama torchrun --nproc_per_node=4 --master_port=20002 ../src/fschat/fastchat/train/train_mem.py \
    --model_name_or_path /data/user_data/jiaruil5/.cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
    --data_path ../data/final_output_full_injected_llama2_train.json \
    --eval_data_path ../data/final_output_full_injected_llama2_val.json \
    --bf16 True \
    --output_dir /data/user_data/jiaruil5/agent/output_5k_qlora_mistral \
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
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb


# deepspeed --num_gpus 6 ~/FastChat/fastchat/train/train_lora.py \

# WANDB_PROJECT=llama torchrun --nproc_per_node=4 --master_port=20002 ../src/fschat/fastchat/train/train_lora.py \
#     --model_name_or_path /data/user_data/jiaruil5/.cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --data_path ../data/final_output_full_injected_llama2_train.json \
#     --eval_data_path ../data/final_output_full_injected_llama2_val.json \
#     --bf16 True \
#     --output_dir /data/user_data/jiaruil5/agent/output_5k_qlora_mistral \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "steps" \
#     --eval_steps 100 \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 4096 \
#     --q_lora True \
#     --flash_attn True \
#     --report_to wandb




# torchrun --nproc_per_node=4 --master_port=20002 ../src/fschat/fastchat/train/train_lora.py \
#     --model_name_or_path /data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --data_path ../data/final_output_full_injected_llama2_train.json \
#     --eval_data_path ../data/final_output_full_injected_llama2_val.json \
#     --bf16 True \
#     --output_dir /data/user_data/jiaruil5/agent/output_5k_qlora_llama \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "steps" \
#     --eval_steps 100 \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 4096 \
#     --q_lora True \
#     --flash_attn True \
#     --report_to wandb

    # --deepspeed ~/FastChat/playground/deepspeed_config_s2.json 
   

#    mistralai/Mistral-7B-Instruct-v0.1	
# codellama/CodeLlama-7b-hf
   
    # --report_to= wandb \


# deepspeed ~/FastChat/fastchat/train/train_lora_t5.py \
#         --model_name_or_path google/flan-t5-xl    \
#         --data_path ~/FastChat/fastchat/data/dummy_conversation.json \
#         --bf16 True \
#         --output_dir output_5k \
#         --num_train_epochs 3 \
#         --per_device_train_batch_size 1 \
#         --per_device_eval_batch_size 1  \
#         --gradient_accumulation_steps 4  \
#         --evaluation_strategy "no"  \
#         --save_strategy "steps"  \
#         --save_steps 300 \
#         --save_total_limit 1 \
#         --learning_rate 2e-5 \
#         --weight_decay 0.     \
#         --warmup_ratio 0.03    \
#         --lr_scheduler_type "cosine"   \
#         --logging_steps 1 \
#         --model_max_length 2048    \
#         --preprocessed_path ~/FastChat/fastchat/preprocessed_data/processed.json \
#         --gradient_checkpointing True \
#         --q_lora True     \
#         --deepspeed ~/FastChat/playground/deepspeed_config_s2.json