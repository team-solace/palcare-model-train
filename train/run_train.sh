# Trained checkpoints is saved to checkpoints
mkdir -p checkpoints
accelerate launch --config_file "palcare-model-train/train/configs/deepspeed/deepspeed_1.yaml" palcare-model-train/train/train_sft.py checkpoints