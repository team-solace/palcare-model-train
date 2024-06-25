import os
from typing import Dict, List

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from jsonargparse import CLI


CPU_COUNT = os.cpu_count()


def process_row(batch: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    return {"count": sum([len(tokenizer(text)['input_ids']) for text in batch])}


def count_dataset(repo_id: str):
    print("Loading Tokenizer...")
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.padding_side = 'right'

    print("Applying Chat Template...")
    train_ds: Dataset = load_dataset("lemousehunter/med-instruct-align-1_parquet", split="train")
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
