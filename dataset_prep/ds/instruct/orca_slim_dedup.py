from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import asdict
from typing import TypedDict, List, Dict

from dataset_prep.custom_type import FormattedFinetuneData

Conversation = TypedDict("Conversation", {"from": str, "value": str})


def process(dataset_row) -> FormattedFinetuneData:
    formatted_conversation = {}

    conversation: List[Conversation] = dataset_row['conversations']

    for msg in conversation:
        if msg["from"] == "system":
            formatted_conversation["system"] = msg["value"]
        elif msg["from"] == "human":
            formatted_conversation["input"] = msg["value"]
        elif msg["from"] == "gpt":
            formatted_conversation["output"] = msg["value"]
        else:
            raise ValueError(f'Invalid message sender \"{msg["from"]}\"')

    return formatted_conversation


def process_dataset(train_ratio: float = 0.9, seed: int = 42) -> DatasetDict:
    ds: Dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="trl_train")

    ds = ds.map(process)

    return ds.train_test_split(test_size=1 - train_ratio, seed=seed)
