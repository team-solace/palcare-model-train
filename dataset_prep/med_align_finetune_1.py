import os

from datasets import Dataset, concatenate_datasets, DatasetDict

from dataset_prep.ds.instruct.wizardlm_evol import process_dataset as process_wizardlm_evol
from dataset_prep.ds.instruct.orca_slim_dedup import process_dataset as process_orca_slim_dedup
from dataset_prep.ds.instruct.natural_instructions import process_dataset as process_natural_instructions
from dataset_prep.ds.instruct.alpaca import process_dataset as process_alpaca
from dataset_prep.ds.med_alignment.medmcqa import process_dataset as process_medmcqa
from dataset_prep.ds.med_alignment.med_calc import process_dataset as process_med_calc

from jsonargparse import CLI


def remove_columns(ds: Dataset):
    # Remove unnecessary columns
    col_names = set(ds.column_names)
    cols_to_keep = {"input", "output", "system"}
    cols_to_remove = list(col_names.difference(cols_to_keep))
    cleaned = ds.remove_columns(cols_to_remove)

    return cleaned


def load_datasets(train_ratio=0.9, seed: int = 7777):
    # Instruct Alignment Datasets
    print("Processing Instruct Datasets...")
    wizardlm_evol: DatasetDict = process_wizardlm_evol(train_ratio, seed)
    orca_slim_dedup: DatasetDict = process_orca_slim_dedup(train_ratio, seed)
    natural_instructions: DatasetDict = process_natural_instructions(train_ratio, seed)
    alpaca: DatasetDict = process_alpaca(train_ratio, seed)

    # Med Alignment Datasets
    print("Processing Med Alignment Datasets...")
    medmcqa: DatasetDict = process_medmcqa(train_ratio, seed)
    med_calc: DatasetDict = process_med_calc(train_ratio, seed)

    # Combine and Shuffle Datasets
    print("Combining Datasets")

    datasets = [wizardlm_evol, orca_slim_dedup, natural_instructions, alpaca, medmcqa, med_calc]
    train_ds_list = [ds['train'] for ds in datasets]
    test_ds_list = [ds['test'] for ds in datasets]

    print("Shuffling Datasets")
    combined_train: Dataset = concatenate_datasets(train_ds_list)
    shuffled_train = combined_train.shuffle(seed=seed)

    combined_test: Dataset = concatenate_datasets(test_ds_list)
    shuffled_test = combined_test.shuffle(seed=seed)

    print("Cleaning Datasets")
    cleaned_train = remove_columns(shuffled_train)
    cleaned_test = remove_columns(shuffled_test)

    cleaned = {
        "train": cleaned_train,
        "test": cleaned_test
    }

    cleaned_dd = DatasetDict(cleaned)

    cleaned_dd.push_to_hub(repo_id="lemousehunter/med-instruct-align-1_parquet", token=os.getenv("HF_WRITE_TOKEN"),
                           commit_message="Part 1 of med-instruct-align dataset")


    print("Done")


if __name__ == "__main__":
    CLI(load_datasets)
