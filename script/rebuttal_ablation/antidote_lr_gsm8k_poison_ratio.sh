#!/bin/bash
#SBATCH -J undercover                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o antidote_lr_gsm8k-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts
lr=1e-4
poison_ratio=0.2
dense_ratio=${1:-0.01}
bad_sample_num=2000
sample_num=5000  
model_path=${4:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of dense ratio is: $dense_ratio"
echo "The value of sample number is: $sample_num"
echo "The learning rate is: $lr"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory





# CUDA_VISIBLE_DEVICES=0 python train.py \
# 	--model_name_or_path ${model_path}\
# 	--lora_folder ckpt/${path_after_slash}_sft \
# 	--data_path PKU-Alignment/BeaverTails_dangerous \
# 	--bf16 True \
# 	--output_dir ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
# 	--num_train_epochs 20 \
# 	--per_device_train_batch_size 5 \
# 	--per_device_eval_batch_size 5 \
# 	--gradient_accumulation_steps 1 \
# 	--save_strategy "steps" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate ${lr} \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0.1 \
# 	--lr_scheduler_type "constant" \
# 	--logging_steps 10 \
# 	--tf32 True \
# 	--eval_steps 10000 \
# 	--cache_dir cache \
# 	--optimizer normal \
# 	--evaluation_strategy  "steps" \
# 	--sample_num $sample_num \
# 	--poison_ratio ${poison_ratio} \
# 	--label_smoothing_factor  0 \
# 	--benign_dataset data/gsm8k.json \

cd poison/evaluation  
echo "Evaluation of after fine-tune model"
# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_sft \
# 	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/poison/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr}


# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/poison/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr}

cd ../../gsm8k
# CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
# 	--lora_folder ../ckpt/${path_after_slash}_sft  \
# 	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr}\
# 	--model_folder ${model_path} \
# 	--output_path ../data/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr}
cd ../

CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}  \
	--lora_folder ckpt/${path_after_slash}_sft  \
	--lora_folder2 ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}_${lr} \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
	--num_train_epochs 0 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--cache_dir cache \
	--optimizer antidote \
	--poison_ratio 1 \
	--sample_num $bad_sample_num \
	--dense_ratio $dense_ratio \
	--benign_dataset data/gsm8k.json \


cd poison/evaluation  

echo "Evaluation of after pruned model" 

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft \
	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr}



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr} \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}_${lr}