#!/bin/bash

#SBATCH --job-name=cllama_full_tplt
#SBATCH --output=cllama_full_tplt.out
#SBATCH --error=cllama_full_tplt.err


# SBATCH --job-name=llama_full_tmp
# SBATCH --output=llama_full_tmp.out
# SBATCH --error=llama_full_tmp.err
#SBATCH --partition=babel-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tianyueo@andrew.cmu.edu



source ~/.bashrc
conda activate agent


# torchrun --nproc_per_node=4 --master_port=20001 ~/FastChat/fastchat/train/train_mem.py \
#     --model_name_or_path /data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
#     --eval_data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_val.json \
#     --bf16 True \
#     --shuffle True \
#     --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_tmp \
#     --num_train_epochs 6 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "steps" \
#     --eval_steps 1 \
#     --save_strategy "epoch" \
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

# torchrun --nproc_per_node=4 --master_port=20003 ~/FastChat/fastchat/train/train_mem.py \
#     --model_name_or_path /data/tir/projects/tir2/users/tianyueo/cllama/models--codellama--CodeLlama-7b-hf/snapshots/bc5283229e2fe411552f55c71657e97edf79066c \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
#     --bf16 True \
#     --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_full_cllama \
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


torchrun --nproc_per_node=4 --master_port=20013 /home/tianyueo/fastchat_jrl/fschat/fastchat/train/train_mistral.py \
    --model_name_or_path /data/tir/projects/tir2/users/tianyueo/cllama/models--codellama--CodeLlama-7b-hf/snapshots/bc5283229e2fe411552f55c71657e97edf79066c \
    --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
    --bf16 True \
    --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_full_cllama_template \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
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
    --template codellama




# torchrun --nproc_per_node=4 --master_port=20013 ~/FastChat/fastchat/train/train_mem.py \
# torchrun --nproc_per_node=4 --master_port=20013 /home/tianyueo/fastchat_jrl/fschat/fastchat/train/train_mistral.py \
# deepspeed /home/tianyueo/fastchat_jrl/fschat/fastchat/train/train_mistral.py \
# deepspeed ~/FastChat/fastchat/train/train_mem.py \
#     --model_name_or_path /data/tir/projects/tir2/users/tianyueo/mistral_flsh_attn/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
#     --bf16 True \
#     --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_full_mistral_flsh_attn_2 \
#     --num_train_epochs 5 \
#     --shuffle True \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --deepspeed ~/FastChat/playground/deepspeed_config_s2.json 
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
        # --eval_steps 1 \
        #     --eval_data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_val.json \





# torchrun --nproc_per_node=4 --master_port=20003 ~/FastChat/fastchat/train/train_mem.py \
#     --model_name_or_path /data/tir/projects/tir2/users/tianyueo/mistral/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
#     --bf16 True \
#     --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_full_mistral \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True


# torchrun --nproc_per_node=4 --master_port=20012 /home/tianyueo/fastchat_jrl/fschat/fastchat/train/train_mistral.py \
#     --model_name_or_path /data/tir/projects/tir2/users/tianyueo/mistral_flsh_attn/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
#     --bf16 True \
#     --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_full_mistral_flsh_attn \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True



# deepspeed --num_gpus 6 ~/FastChat/fastchat/train/train_lora.py \
# torchrun --nproc_per_node=4 --master_port=20002 ~/FastChat/fastchat/train/train_lora.py \
#     --model_name_or_path /data/tir/projects/tir2/users/tianyueo/mistral_flsh_attn/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --data_path /home/tianyueo/agent/data/final_output_full_injected_llama2_train.json \
#     --bf16 True \
#     --output_dir /data/tir/projects/tir2/users/tianyueo/output_5k_qlora_cllama \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \ 
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
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
#     --gradient_checkpointing True
    
    # --deepspeed ~/FastChat/playground/deepspeed_config_s2.json 


# mistralai/Mistral-7B-Instruct-v0.1	
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