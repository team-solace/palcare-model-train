#!/bin/bash

export DOWNLOAD_DIR=/mnt/checkpoints
export CHECKPOINT_DIR=/mnt/checkpoints/lemousehunter/epflllm_meditron-7b-base
export SAVE_DIR=/mnt/out_full

# echo "Downloading and converting model"
# litgpt download --checkpoint_dir ${DOWNLOAD_DIR} lemousehunter/epflllm_meditron-7b-base
wandb login ${WANDB_API_KEY}

# echo "Copying correct model config"
# cp ${MODEL_CONFIG_PATH} ${CHECKPOINT_DIR}/model_config.yaml
## litgpt convert_to_litgpt checkpoints/lemousehunter/epflllm_meditron-7b-base
# litgpt finetune_lora --devices 4 --lora_r 32 --lora_alpha 16 --train.max_seq_length 4096 --lora_dropout 0.05 --lora_query true --lora_value true --train.lr_warmup_steps 10 --train.global_batch_size 24 --train.micro_batch_size 6 --data MedInstructAlign --precision bf16-true --train.save_interval 200 --eval.interval 200 --logger_name wandb --out_dir ${SAVE_DIR} ${CHECKPOINT_DIR}
# litgpt finetune_adapter_v2 --out_dir ${SAVE_DIR} --devices 4 --train.max_seq_length 4096 --train.lr_warmup_steps 10 --train.global_batch_size 72 --train.micro_batch_size 18 --data MedInstructAlign --precision bf16-true --train.save_interval 200 --eval.interval 200 --logger_name wandb ${CHECKPOINT_DIR}
# echo "Launching litgpt finetune_full"
litgpt finetune_full --config full.yaml

# litgpt finetune_lora --devices 2 --data MedInstructAlign --train.max_seq_length 4096 --logger_name wandb --quantize bnb.int8-training --precision 16-true checkpoints/lemousehunter/epflllm_meditron-7b-base