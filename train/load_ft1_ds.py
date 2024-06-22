from typing import TypedDict, List

from datasets import load_dataset, Dataset


class Finetune1DatasetRow(TypedDict):
    input: str
    output: str
    system: str


class ChatMessage(TypedDict):
    role: str
    content: str


class FormattedDatasetRow(TypedDict):
    messages: List[ChatMessage, ChatMessage, ChatMessage]


def process(dataset_row) -> FormattedDatasetRow:
    formatted_row: FormattedDatasetRow = dict(messages=[])

    for col in dataset_row:
        formatted_row['messages'].append({"role": col, "content": dataset_row[col]})

    return formatted_row


def load_finetune_1() -> (Dataset, Dataset):
    train_ds: Dataset = load_dataset("lemousehunter/med-instruct-align-1_parquet")['train']
    assert train_ds.column_names == ["input", "output", "system"]
    train_ds = train_ds.map(process)

    val_ds: Dataset = load_dataset("lemousehunter/med-instruct-align-1_parquet")['test']
    assert val_ds.column_names == ["input", "output", "system"]
    val_ds = val_ds.map(process)

    return train_ds, val_ds