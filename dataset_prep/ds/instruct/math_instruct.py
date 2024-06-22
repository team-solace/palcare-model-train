from datasets import load_dataset, Dataset, DatasetDict
from typing import TypedDict


def process(dataset_row):
    dataset_row['system'] = "You are an expert in the Medical Field, and you are tasked with performing the following calculations. Please explain your reasoning for each step before deriving your answer."

    return dataset_row


def process_dataset(train_ratio: int = 0.9, seed: int = 42) -> DatasetDict:
    ds: Dataset = load_dataset("TIGER-Lab/MathInstruct", split="train")

    ds = ds.rename_columns({"instruction": "input"})

    ds = ds.map()

    return ds.train_test_split(test_size=1 - train_ratio, seed=seed)
