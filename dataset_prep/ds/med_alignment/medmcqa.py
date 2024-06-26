from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from dataclasses import asdict

from dataset_prep.custom_type import FormattedFinetuneData


def process(dataset_row) -> FormattedFinetuneData:
    formatted_qa = {}

    answer_mapping = {
        0: "opa",
        1: "opb",
        2: "opc",
        3: "opd",
        "0": "opa",
        "1": "opb",
        "2": "opc",
        "3": "opd",
    }

    formatted_qa["system"] = f"You are an expert in the medical field, and will be tasked to answer the following question with explanations. For each question, you will be given 4 options to choose a correct answer from. The question will be on {dataset_row['subject_name']}, specifically, the {dataset_row['topic_name']}."

    formatted_qa["input"] = f'{dataset_row["question"]} {dataset_row["opa"]}, {dataset_row["opb"]}, {dataset_row["opc"]} or {dataset_row["opd"]}?'

    try:
        correct_ans = dataset_row["cop"]
        answer_col = answer_mapping.get(correct_ans)
        correct_answer = dataset_row[answer_col]
    except KeyError:
        raise KeyError(f"Answer column {answer_col} not found in dataset row: {dataset_row}. Please check if answer {correct_ans} is an expected value.")

    explanation = dataset_row["exp"]

    formatted_qa["output"] = f'The correct answer is {correct_answer}. The explanation is as follows:\n{explanation}'

    return formatted_qa


def process_dataset(train_ratio: float = 0.9, seed: int = 42) -> DatasetDict:
    train: Dataset = load_dataset("openlifescienceai/medmcqa", split="trl_train")
    val: Dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    # test: Dataset = load_dataset("openlifescienceai/medmcqa", split="test")

    ds: Dataset = concatenate_datasets([train, val])

    ds = ds.map(process)

    return ds.train_test_split(test_size=1 - train_ratio, seed=seed)
