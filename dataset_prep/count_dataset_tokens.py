import os
from typing import Dict, List

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from jsonargparse import CLI
from train.load_ft1_ds import process

CPU_COUNT = os.cpu_count()


def process_row(batch: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    return {"count": sum([len(tokenizer(text)['input_ids']) for text in batch])}


def count_dataset(dataset_repo_id: str, tokenizer_repo_id: str):
    print("Loading Tokenizer...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_repo_id)
    tokenizer.padding_side = 'right'

    print("Loading Dataset...")
    train_ds: Dataset = load_dataset(dataset_repo_id, split="train")

    print("Applying Chat Template...")

    train_ds = train_ds.map(process)
    train_ds = train_ds.map(lambda x: {
        "messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)
    }, num_proc=CPU_COUNT)

    print("Counting tokens...")
    count_ds: Dataset = train_ds.map(lambda batch: process_row(batch, tokenizer), batched=True, batch_size=8,
                                     num_proc=CPU_COUNT)

    print("Processing Total...")
    count_df: pd.DataFrame = count_ds.to_pandas()
    count: int = count_df["count"].sum()

    print("Train dataset has", count, "tokens")


if __name__ == "__main__":
    CLI(count_dataset)
