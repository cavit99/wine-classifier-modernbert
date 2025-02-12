import pandas as pd

def analyze_distribution(combined_df, column, display_name=None, top_n=20):
    """
    Analyzes the combined distribution of a given column across a DataFrame.

    Args:
        combined_df (pd.DataFrame): The concatenated DataFrame of all CSV files.
        column (str): The column name to analyze.
        display_name (str, optional): Optional display name for the column. 
                                      If not provided, column's capitalized name is used.
        top_n (int): Number of top items to display. Defaults to 20.
    """
    if display_name is None:
        display_name = column.capitalize()
    
    counts = combined_df[column].value_counts()

    print(f"\nCombined {display_name} Distribution Analysis:")
    print("-" * 40)

    # Total number of unique items
    num_unique = len(counts)
    print(f"Total Unique {display_name}s: {num_unique}")

    # Total number of entries
    total_entries = counts.sum()
    print(f"Total Number of Entries: {total_entries}")

    # Top N items
    print(f"\nTop {top_n} {display_name}s:")
    print(counts.head(top_n))

    # "Other" items (if applicable)
    if len(counts) > top_n:
        other_count = len(counts) - top_n
        other_sum = counts.iloc[top_n:].sum()
        print(f"\n... and {other_count} other {display_name}s with a combined total of {other_sum} entries.")

    # Percentiles (including some more extreme ones)
    percentiles = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    percentile_values = counts.quantile(percentiles)
    print(f"\n{display_name} Count Percentiles:")
    for p, val in percentile_values.items():
        print(f"  {p:.0%}: {int(val)}") 

    # Calculate and print the number of items that make up the top 50% of entries
    cumulative_sum = counts.cumsum()
    top_50_percent_cutoff = total_entries * 0.5
    num_items_top_50 = (cumulative_sum <= top_50_percent_cutoff).sum()
    if num_items_top_50 == 0:
        # Handle edge case where the most frequent item is >50%
        num_items_top_50 = 1
    print(f"\nNumber of {display_name}s that constitute the top 50% of entries: {num_items_top_50}")

    # Calculate and print the percentage of items that appear only once
    single_occurrence_items = (counts == 1).sum()
    single_occurrence_percentage = (single_occurrence_items / num_unique) * 100
    print(f"Percentage of {display_name}s that appear only once: {single_occurrence_percentage:.2f}%")

    # Calculate and print the percentage of entries that are in the top 1% of items
    top_1_percent_cutoff = counts.quantile(0.99)
    top_1_percent_entries = counts[counts >= top_1_percent_cutoff].sum()
    top_1_percent_entries_percentage = (top_1_percent_entries / total_entries) * 100
    print(f"Percentage of entries that are in the top 1% of {display_name}s: {top_1_percent_entries_percentage:.2f}%")

def analyze_variety_distribution(file_paths):
    """
    Analyzes the combined distribution of wine varieties and countries across multiple CSV files.
    
    This function:
      - Loads each CSV file (with error handling)
      - Combines them
      - Prints the simple country distribution
      - Analyzes both 'variety' and 'country' columns

    Args:
        file_paths (list): A list of paths to the CSV files.
    
    Returns:
        pd.DataFrame: The combined DataFrame if successful, else None.
    """
    all_dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File is empty: {file_path}")
            return None
        except pd.errors.ParserError:
            print(f"Error: Could not parse CSV file: {file_path}")
            return None

    if not all_dfs:
        print("Error: No valid CSV files were loaded.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("Country Distribution:")
    print(combined_df['country'].value_counts())
    
    analyze_distribution(combined_df, 'variety', display_name='Variety')
    analyze_distribution(combined_df, 'country', display_name='Country')
    
    return combined_df

def analyze_token_length_distribution(df, tokenizer, column="text", max_length=256):
    """
    Analyzes the token length distribution for a given column using the provided tokenizer.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        tokenizer: HuggingFace tokenizer object.
        column (str): Column name to analyze. Defaults to "text".
        max_length (int): The max_length used during tokenization.
        
    This function prints:
      - Total record count
      - Minimum token length
      - Maximum token length
      - Mean token length
      - Median token length
      - A distribution (quantiles) of token counts
      - The percentage of records that exceed the max_length 
    """
    token_lengths = df[column].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=True)))
    
    print(f"\nToken Length Distribution for column '{column}':")
    print("-" * 60)
    print(f"Total records: {len(token_lengths)}")
    print(f"Minimum tokens: {token_lengths.min()}")
    print(f"Maximum tokens: {token_lengths.max()}")
    print(f"Mean tokens: {token_lengths.mean():.2f}")
    print(f"Median tokens: {token_lengths.median()}")
    
    # Detailed quantile distribution.
    quantiles = token_lengths.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    print("\nQuantiles:")
    for quantile, value in quantiles.items():
        print(f"  {quantile:.0%}: {value:.0f}")
    
    # Calculate percentage of records that exceed the max token length.
    exceeding_pct = (token_lengths > max_length).mean() * 100
    print(f"\nPercentage of sequences exceeding max_length ({max_length} tokens): {exceeding_pct:.2f}%")

if __name__ == '__main__':
    from transformers import AutoTokenizer

    files = ['dataset/winemag-data_first150k.csv', 'dataset/winemag-data-130k-v2.csv']

    # Analyze variety and country distribution.
    combined_df = analyze_variety_distribution(files)
    if combined_df is None:
        exit(1)

    # Create the 'text' field by concatenating 'country' and 'description'
    if 'country' in combined_df.columns and 'description' in combined_df.columns:
        combined_df["text"] = combined_df["country"].str.strip() + " [SEP] " + combined_df["description"].str.strip()
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        analyze_token_length_distribution(combined_df, tokenizer, column="text", max_length=256)
    else:
        print("Error: The DataFrame must contain 'country' and 'description' columns for token analysis.")