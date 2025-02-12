import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting
import seaborn as sns  # Import seaborn

def analyze_variety_distribution(file_paths):
    """
    Analyzes the combined distribution of wine varieties across multiple CSV files.

    Args:
        file_paths (list): A list of paths to the CSV files.
    """

    all_dfs = []
    for file_path in file_paths:
        try:  # Use try-except to handle potential file errors
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return  # Exit the function if a file is not found
        except pd.errors.EmptyDataError:
            print(f"Error: File is empty: {file_path}")
            return
        except pd.errors.ParserError:
            print(f"Error: Could not parse CSV file: {file_path}")
            return

    if not all_dfs:  # Check if the list of dataframes is empty
        print("Error: No valid CSV files were loaded.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    variety_counts = combined_df['variety'].value_counts()

    print("\nCombined Variety Distribution Analysis:")
    print("-" * 40)

    # Total number of unique varieties
    num_unique_varieties = len(variety_counts)
    print(f"Total Unique Varieties: {num_unique_varieties}")

    # Total number of entries
    total_entries = variety_counts.sum()
    print(f"Total Number of Wine Entries: {total_entries}")

    # Top N varieties (you can still customize this)
    top_n = 20
    print(f"\nTop {top_n} Varieties:")
    print(variety_counts.head(top_n))

    # "Other" varieties
    if len(variety_counts) > top_n:
        other_count = len(variety_counts) - top_n
        other_sum = variety_counts.iloc[top_n:].sum()
        print(f"\n... and {other_count} other varieties with a combined total of {other_sum} entries.")

    # Percentiles (including some more extreme ones)
    percentiles = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    percentile_values = variety_counts.quantile(percentiles)
    print("\nVariety Count Percentiles:")
    for p, val in percentile_values.items():
        print(f"  {p:.0%}: {int(val)}")  # Format as percentage and integer

    # Calculate and print the number of varieties that make up the top 50% of entries
    cumulative_sum = variety_counts.cumsum()
    top_50_percent_cutoff = total_entries * 0.5
    num_varieties_top_50 = (cumulative_sum <= top_50_percent_cutoff).sum()
    if num_varieties_top_50 == 0:
        # Handle edge case where the most frequent variety is >50%
        num_varieties_top_50 = 1
    print(f"\nNumber of varieties that constitute the top 50% of entries: {num_varieties_top_50}")

    # Calculate and print the percentage of varieties that appear only once
    single_occurrence_varieties = (variety_counts == 1).sum()
    single_occurrence_percentage = (single_occurrence_varieties / num_unique_varieties) * 100
    print(f"Percentage of varieties that appear only once: {single_occurrence_percentage:.2f}%")

    # Calculate and print the percentage of entries that are in the top 1% of varieties
    top_1_percent_cutoff = variety_counts.quantile(0.99)
    top_1_percent_entries = variety_counts[variety_counts >= top_1_percent_cutoff].sum()
    top_1_percent_entries_percentage = (top_1_percent_entries / total_entries) * 100
    print(f"Percentage of entries that are in the top 1% of varieties: {top_1_percent_entries_percentage:.2f}%")

if __name__ == '__main__':
    files = ['dataset/winemag-data_first150k.csv', 'dataset/winemag-data-130k-v2.csv']
    analyze_variety_distribution(files)