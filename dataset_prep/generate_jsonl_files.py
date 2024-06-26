from datasets import load_dataset, Dataset, DatasetDict
from jsonargparse import CLI


def generate_json_files(file_dir: str, repo_id: str, has_split: bool = True):
    print("loading dataset")

    if not has_split:
        ds: Dataset = load_dataset(repo_id, split="trl_train")
        print("Splitting Dataset")
        train_test: DatasetDict = ds.train_test_split(test_size=0.1, seed=42)

    else:
        train_test: DatasetDict = load_dataset(repo_id)

    print("Saving trl_train to JSONL")
    train_test['trl_train'].to_json(f"{file_dir}/trl_train.jsonl", orient="records", lines=True)

    print("Saving test to JSONL")
    train_test['test'].to_json(f"{file_dir}/val.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    CLI(generate_json_files)