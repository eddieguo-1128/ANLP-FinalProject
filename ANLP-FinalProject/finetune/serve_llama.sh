#!/bin/bash
#SBATCH --job-name=server_llama
#SBATCH --output=server_llama.out
#SBATCH --error=server_llama.err
#SBATCH --partition=babel-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate agent



MODEL_DIR="/home/tianyueo/agent/fine_tune/output_5k"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=1528 \
    --model="$MODEL_DIR" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=4096

# MODEL_DIR="/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf/"
# test -d "$MODEL_DIR"
# python -O -u -m vllm.entrypoints.openai.api_server \
#     --port=1527 \
#     --model=/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf/ \
#     --tokenizer=hf-internal-testing/llama-tokenizer \
#     --tensor-parallel-size=4






