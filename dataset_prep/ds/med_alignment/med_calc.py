from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from dataclasses import asdict
from typing import TypedDict, List, Dict

from dataset_prep.custom_type import FormattedFinetuneData


def process(dataset_row) -> FormattedFinetuneData:
    formatted_qa = {}

    formatted_qa["system"] = "You are an expert in the medical field. You are tasked to answer a medical question by making some calculations using the Patient Note given. You are to first extract the relevant entities explain your calculations used to derive your answer in a step by step manner."
    formatted_qa["input"] = f'{dataset_row["Question"]}\nPatient Note:\n{dataset_row["Patient Note"]}'
    formatted_qa['output'] = f'The type of calculations required to answer this question is {dataset_row["Calculator Name"]}.\n\nRelevant Entities:\n{dataset_row["Relevant Entities"]}\n\nAnswer:\n{dataset_row["Ground Truth Answer"]}\n\nExplanation:\n{dataset_row["Ground Truth Explanation"]}'

    return formatted_qa


def process_dataset(train_ratio: float = 0.9, seed: int = 42) -> DatasetDict:
    train: Dataset = load_dataset("ncbi/MedCalc-Bench", split="train")
    test: Dataset = load_dataset("ncbi/MedCalc-Bench", split="test")

    ds: Dataset = concatenate_datasets([train, test])

    ds = ds.map(process)

    return ds.train_test_split(test_size=1-train_ratio, seed=seed)
