from datasets import load_dataset, Dataset


def load_finetune_1():
    train_ds: Dataset = load_dataset("lemousehunter/med-instruct-align-1_parquet")['train']
    assert train_ds.column_names == ["input", "output", "system"]

    split = train_ds.train_test_split(test_size=0.04)
    train_1_ds = split['train']
    train_2_ds = split['test']

    train_1_ds.push_to_hub("lemousehunter/med-instruct-align-1_train_293K")
    train_2_ds.push_to_hub("lemousehunter/med-instruct-align-1_train_7.6M")


if __name__ == "__main__":
    load_finetune_1()