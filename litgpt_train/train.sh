#!/bin/bash

litgpt download lemousehunter/epflllm_meditron-7b-base
wandb login ${WANDB_API_KEY}
litgpt finetune_lora --devices 2 --data MedInstructAlign --logger_name wandb checkpoints/lemousehunter/epflllm_meditron-7b-base
