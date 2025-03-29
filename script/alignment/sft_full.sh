#!/bin/bash
#SBATCH -J sft_full                 # Job name
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=30G
#SBATCH -o sft-%j.out                         # Combined output and error messages file

# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

model_path=${1:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of sample number is: $sample_num"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory





# CUDA_VISIBLE_DEVICES=0 python train.py \
# 	--model_name_or_path ${model_path} \
# 	--data_path PKU-Alignment/BeaverTails_safe \
# 	--bf16 True \
# 	--output_dir ckpt/${path_after_slash}_sft_full \
# 	--num_train_epochs 10 \
# 	--per_device_train_batch_size 5 \
# 	--per_device_eval_batch_size 5 \
# 	--gradient_accumulation_steps 1 \
# 	--evaluation_strategy "no" \
# 	--save_strategy "steps" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate  1e-5 \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0.1 \
# 	--lr_scheduler_type "cosine" \
# 	--logging_steps 1 \
# 	--tf32 True \
# 	--cache_dir cache \
# 	--optimizer sft \
# 	--sample_num 5000 \
# 	--full_finetuning True

cd poison/evaluation  

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--model_folder ../../ckpt/${path_after_slash}_sft_full \
	--output_path ../../data/poison/${path_after_slash}_sft_full

CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_sft_full