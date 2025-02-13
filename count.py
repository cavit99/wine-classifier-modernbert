import numpy as np
from collections import Counter
from datasets import load_dataset

def analyze_distribution(dataset, column, display_name=None, top_n=20):
    """
    Analyzes the distribution of a given column from a Hugging Face dataset.

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset containing the reviews.
        column (str): The column name to analyze.
        display_name (str, optional): Optional display name for the column. 
                                      If not provided, the column's capitalized name is used.
        top_n (int): Number of top items to display. Defaults to 20.
    """
    if display_name is None:
        display_name = column.capitalize()

    # Extract column values (assumes dataset[column] returns a list)
    values = dataset[column]
    counts = Counter(values)
    sorted_counts = counts.most_common()  # Sorted from most common to least

    print(f"\nCombined {display_name} Distribution Analysis:")
    print("-" * 40)

    # Total number of unique items
    num_unique = len(counts)
    print(f"Total Unique {display_name}s: {num_unique}")

    # Total number of entries
    total_entries = sum(counts.values())
    print(f"Total Number of Entries: {total_entries}")

    # Top N items
    print(f"\nTop {top_n} {display_name}s:")
    for key, val in sorted_counts[:top_n]:
        print(f"{key}: {val}")

    # "Other" items (if applicable)
    if len(sorted_counts) > top_n:
        other_count = len(sorted_counts) - top_n
        other_sum = sum(val for _, val in sorted_counts[top_n:])
        print(f"\n... and {other_count} other {display_name}s with a combined total of {other_sum} entries.")

    # Percentiles (including some more extreme ones)
    percentiles = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    freq_values = np.array(list(counts.values()))
    q_values = np.quantile(freq_values, percentiles)
    print(f"\n{display_name} Count Percentiles:")
    for p, val in zip(percentiles, q_values):
        print(f"  {p:.0%}: {int(val)}")

    # Calculate and print the number of items that make up the top 50% of entries
    cumulative_sum = 0
    num_items_top_50 = 0
    for _, val in sorted_counts:
        cumulative_sum += val
        num_items_top_50 += 1
        if cumulative_sum >= total_entries * 0.5:
            break
    if num_items_top_50 == 0:
        num_items_top_50 = 1
    print(f"\nNumber of {display_name}s that constitute the top 50% of entries: {num_items_top_50}")

    # Calculate and print the percentage of items that appear only once
    single_occurrence_items = sum(1 for v in counts.values() if v == 1)
    single_occurrence_percentage = (single_occurrence_items / num_unique) * 100
    print(f"Percentage of {display_name}s that appear only once: {single_occurrence_percentage:.2f}%")

    # Calculate and print the percentage of entries that are in the top 1% of items
    top_1_percent_cutoff = np.quantile(freq_values, 0.99)
    top_1_percent_entries = sum(val for val in counts.values() if val >= top_1_percent_cutoff)
    top_1_percent_entries_percentage = (top_1_percent_entries / total_entries) * 100
    print(f"Percentage of entries that are in the top 1% of {display_name}s: {top_1_percent_entries_percentage:.2f}%")

def analyze_variety_distribution():
    """
    Loads the 'spawn99/wine-reviews' dataset from Hugging Face and analyzes the distribution
    of wine varieties. It also extracts and prints a list of varieties that include a hyphen, forward slash,
    or backslash (e.g., 'cabarnet-syrah') along with how many times they appear, and provides a separate
    distribution analysis for these entries. Additionally, it extracts and prints the list of wine varieties
    containing 'blend'.
    
    Returns:
        datasets.Dataset: The loaded dataset if successful, else None.
    """
    try:
        ds = load_dataset("spawn99/wine-reviews")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # If the dataset has multiple splits (e.g., a DatasetDict), choose 'train' if available.
    if isinstance(ds, dict):
        if "train" in ds:
            dataset = ds["train"]
        else:
            dataset = list(ds.values())[0]
    else:
        dataset = ds

    # Analyze the 'variety' column from the dataset.
    analyze_distribution(dataset, 'variety', display_name='Variety')

    # Define special characters to look for.
    special_chars = ['-', '/', '\\']
    
    # Compute a counter for varieties containing any of the special characters (filtering out None values)
    special_char_counter = Counter(
        var for var in dataset['variety']
        if var is not None and any(char in var for char in special_chars)
    )

    # Show the counts (unique entries with their frequency)
    print("\nCounts for wine varieties containing '-', '/' or '\\':")
    for variety, freq in special_char_counter.most_common():
        print(f"{variety}: {freq}")
    
    # Display a separate distribution analysis for these special-character varieties.
    print("\nDetailed Distribution for wine varieties containing '-', '/' or '\\':")
    print("-" * 40)
    total_special = sum(special_char_counter.values())
    num_unique_special = len(special_char_counter)
    print(f"Total Unique Special-Character Varieties: {num_unique_special}")
    print(f"Total Occurrences of Special-Character Varieties: {total_special}")
    
    if num_unique_special > 0:
        special_freq_values = np.array(list(special_char_counter.values()))
        percentiles = [0, 0.25, 0.5, 0.75, 1]
        q_values = np.quantile(special_freq_values, percentiles)
        print("\nCount Percentiles for Special-Character Varieties:")
        for p, q in zip(percentiles, q_values):
            print(f"  {p*100:.0f}th percentile: {int(q)}")
    else:
        print("No special-character varieties found for distribution analysis.")
    
    # Extract and print the unique list of wine varieties that contain 'blend'.
    blend_varieties = [
        var for var in dataset['variety']
        if var is not None and 'blend' in var.lower()
    ]
    unique_blend_varieties = sorted(set(blend_varieties))
    print("\nUnique wine varieties containing 'blend':")
    for variety in unique_blend_varieties:
        print(variety)
    
    return dataset

if __name__ == '__main__':
    dataset = analyze_variety_distribution()
    if dataset is None:
        exit(1)
