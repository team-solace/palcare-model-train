from datasets import load_dataset, Dataset
from jsonargparse import CLI


def load_finetune_1(hf_write_token: str):
    train_ds: Dataset = load_dataset("lemousehunter/med-instruct-align-1_parquet")['trl_train']
    assert train_ds.column_names == ["input", "output", "system"]

    split = train_ds.train_test_split(test_size=0.04)
    train_1_ds = split['trl_train']
    train_2_ds = split['test']

    train_1_ds.push_to_hub("lemousehunter/med-instruct-align-1_train_293K", token=hf_write_token)
    train_2_ds.push_to_hub("lemousehunter/med-instruct-align-1_train_7.6M", token=hf_write_token)


if __name__ == "__main__":
    CLI(load_finetune_1)