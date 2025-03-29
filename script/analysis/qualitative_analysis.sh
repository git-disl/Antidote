#!/bin/bash
#SBATCH -J undercover                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o qualitative-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

lr=${1:-1e-3}
poison_ratio=0
dense_ratio=0.01
bad_sample_num=2000
sample_num=5000  
model_path=${4:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 

cd ../../poison/evaluation  

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft \
	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
    --num_test_data 5


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr}

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft \
	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
    --num_test_data 5


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr}




# cd ../../gsm8k

# CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
# 	--lora_folder ../ckpt/${path_after_slash}_sft  \
# 	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
# 	--model_folder ${model_path} \
# 	--output_path ../data/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
#     --num_test_data 5


# CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
# 	--lora_folder ../ckpt/${path_after_slash}_sft  \
# 	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
# 	--model_folder ${model_path} \
# 	--output_path ../data/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
#     --num_test_data 5