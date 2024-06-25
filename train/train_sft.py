import os

from accelerate import Accelerator
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig
from train.load_ft1_ds import load_finetune_1
from jsonargparse import CLI
# from unsloth import FastLanguageModel

model_id = "epfl-llm/meditron-7b"


def train(checkpoint_dir: str):
    # Load Dataset
    print("Loading Dataset")
    train_ds, val_ds = load_finetune_1()

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    # Load Model
    print("Loading Model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    # Load Tokenizer
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    # Instruct-Format Dataset
    print("Instruct Formatting Dataset")
    train_ds = train_ds.map(lambda x: {
        "messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})
    val_ds = val_ds.map(lambda x: {
        "messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    print("Loading LoRA Config")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    print("Loading SFT Config")
    peft_output_dir = f"{checkpoint_dir}/lemousehunter/meditron-7b-medalign"
    args = SFTConfig(
        hub_token=os.environ['HF_WRITE_TOKEN'],  # push to hub token
        output_dir=peft_output_dir,  # directory to save and repository id
        num_train_epochs=5,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size per device during training
        eval_accumulation_steps=1,  # number of eval steps to accumulate before evaluation
        adam_beta1=0.9,  # beta1 for adam optimizer
        adam_beta2=0.95,  # beta1 for adam optimizer
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch",  # use ##fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=True,  # push model to hub
        report_to="wandb",  # report metrics to wandb
        weight_decay=0,
        dataset_text_field="messages",
        ddp_timeout=3600,
        do_eval=True,
        run_name="meditron-7b-medalign_debug",
        eval_strategy="steps",
    )

    max_seq_length = 4096  # max sequence length for model and packing of the dataset

    print("Loading SFT Trainer")
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )

    print("Starting Training")
    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    print("Saved Model")
    # save model
    trainer.save_model()

    # Load Model with PEFT adapter
    final_model = AutoPeftModelForCausalLM.from_pretrained(
        peft_output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    final_model.save_pretrained(peft_output_dir)


if __name__ == "__main__":
    CLI(train)
