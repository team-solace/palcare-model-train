#!/bin/bash

litgpt download lemousehunter/epflllm_meditron-7b-base
wandb login ${WANDB_API_KEY}
# litgpt convert_to_litgpt checkpoints/lemousehunter/epflllm_meditron-7b-base
litgpt finetune_lora --devices 2 --lora_r 32 --lora_alpha 16 --lora_dropout 0.05 --lora_query true --lora_value true --train.lr_warmup_steps 10 --train.global_batch_size 16 --train.micro_batch_size 1 --data MedInstructAlign --train.max_seq_length 4096 --precision bf16-true --logger_name wandb checkpoints/lemousehunter/epflllm_meditron-7b-base
# litgpt finetune_lora --devices 2 --data MedInstructAlign --train.max_seq_length 4096 --logger_name wandb --quantize bnb.int8-training --precision 16-true checkpoints/lemousehunter/epflllm_meditron-7b-base
