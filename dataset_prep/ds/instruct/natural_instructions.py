from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from typing import TypedDict


def process_dataset(train_ratio: float = 0.9, seed: int = 42) -> DatasetDict:
    train: Dataset = load_dataset("Muennighoff/natural-instructions", split="trl_train")
    val: Dataset = load_dataset("Muennighoff/natural-instructions", split="validation")
    test: Dataset = load_dataset("Muennighoff/natural-instructions", split="test")

    ds: Dataset = concatenate_datasets([train, val, test])

    ds = ds.rename_columns({"definition": "system", "inputs": "input", "targets": "output"})

    ds = ds.remove_columns(["task_name", "id"])

    return ds.train_test_split(test_size=1 - train_ratio, seed=seed)
