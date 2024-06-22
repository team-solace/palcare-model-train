from datasets import load_dataset, Dataset, DatasetDict
from typing import TypedDict


def process_dataset(train_ratio: float = 0.9, seed: int = 42) -> DatasetDict:
    ds: Dataset = load_dataset("alexl83/AlpacaDataCleaned", split="train")

    ds = ds.rename_columns({"instruction": "system"})

    return ds.train_test_split(test_size=1 - train_ratio, seed=seed)
