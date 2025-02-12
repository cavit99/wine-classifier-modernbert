#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

def preprocess_dataset():
    # Load raw CSV data without dropping any columns, specifying the index column
    df1 = pd.read_csv("dataset/winemag-data_first150k.csv", index_col=0)
    df2 = pd.read_csv("dataset/winemag-data-130k-v2.csv", index_col=0)

    # Combine the two dataframes preserving all original columns
    df = pd.concat([df1, df2], ignore_index=True)

    # Create a Hugging Face Dataset from the Pandas DataFrame
    full_dataset = Dataset.from_pandas(df)

    # Use native dataset splitting to produce train, validation, and test splits.
    # Desired split: train = 70%, validation = 10%, test = 20%
    # First, split off 30% of the data.
    split_ds = full_dataset.train_test_split(test_size=0.3, seed=42)
    train_ds = split_ds["train"]
    temp_ds = split_ds["test"]

    # Further split the temporary set into validation (10% overall) and test (20% overall)
    # Since temp_ds represents 30% of the overall data, splitting with test_size=2/3 yields:
    # validation ≈ 10% and test ≈ 20%
    temp_split = temp_ds.train_test_split(test_size=2/3, seed=42)
    validation_ds = temp_split["train"]
    test_ds = temp_split["test"]

    # Create a DatasetDict with the splits
    dataset = DatasetDict({
        "train": train_ds,
        "validation": validation_ds,
        "test": test_ds
    })

    return dataset

def main():
    dataset = preprocess_dataset()

    # Push the dataset to the Hugging Face Hub
    repo_id = "spawn99/wine-reviews"  # Change to your desired Hub repository name.
    dataset.push_to_hub(repo_id, private=False)
    print(f"Processed dataset successfully pushed to Hugging Face Hub at '{repo_id}'.")

if __name__ == "__main__":
    main()