<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning </h1>



Antidote is a post-fine-tuning safety alignment method against the threat of harmful fine-tuning. We consider a three-stage scheme for safety-aligned fine-tuning-as-a-service: 

i) **Alignment stage**, in which we align the model with human-preference dataset (alignment dataset).

ii) **User fine-tuning stage**, in which we finetune the model with a user finetuning dataset (which is mixed with harmful instance). 

iii) **Post fine-tuning stage**, in which Antidote is applied. The idea is to remove harmful parameters to repair the model from harmful behavior.    



## Main code logistic
We implement a cusomized trainer on top of the original HuggingFace Trainer. To achieve Antidote,  we append a function `save_mask()` in `AntidoteTrainer`. This fuction calls the wanda score calculation function as follows, and derives the harmful mask that captures the topk important parameters over the realignment dataset.   

`self.mask = prune_wanda_outlier(self.args, self.model.model, None, device=torch.device("cuda:0"))`




## Package requirement
The package requirement is listed in `antidote.yml` and `antidote_pip.txt`. Run the following code to install the packages with anaconda and pip.  
```
conda env create -f antidote.yml
pip install -r antidote_pip.txt
```

## Data  preparation
For finetuning task, we first need to run the following scripts to prepare the sueprvised finetuning data.
```
cd sst2
python build_dataset.py
cd ../gsm8k
python build_dataset.py
cd ../ag_news
python build_dataset.py
cd ..
```

## Huggingface Llama2 access
Llama2-7B is a gated repo, which need a formal request to get access to the model. Check out https://huggingface.co/meta-llama/Llama-2-7b-hf.
After applying permission from meta, you should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.



## Example command to run

We prepare scripts for re-producing all the experiments in the paper. We recommend to use Slurm to reproduce the results as the logging file will be automatically organized into the script directory (if you don't use Slurm, just replace `sbatch` with `bash` in our example).

We first run SFT to produce the aligned model. 
```
cd script/alignment
sbatch  SFT.sh
```
Then we run the fine-tuning stage/post-fine-tuning stage by calling this script:
```
cd ../finetune
sbatch  antidote_poison_ratio.sh 0.1
```


For comparison, we can finetune the model with SFT in the same data setting.

```
sbatch  sft_poison_ratio.sh 0.1
cd ../..
```








