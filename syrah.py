from datasets import load_dataset

def check_syrah_distribution(repo_id: str = "spawn99/wine-reviews"):
    """
    Loads a dataset from the Hugging Face Hub and checks the distribution
    of the 'Syrah' variety across its splits.

    Args:
        repo_id: The Hugging Face Hub repository ID.
    """

    try:
        dataset = load_dataset(repo_id)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    for split in dataset.keys():
        print(f"Checking Syrah distribution in {split} split:")
        syrah_count = 0
        total_count = 0

        # Iterate through the examples in the split
        for example in dataset[split]:
            total_count += 1
            if example['variety'] == 'Syrah':
                syrah_count += 1

        print(f"  Total examples: {total_count}")
        print(f"  Syrah examples: {syrah_count}")
        if total_count > 0:  # Avoid division by zero
            print(f"  Syrah percentage: {syrah_count / total_count * 100:.2f}%")
        else:
            print("  Split is empty.")
        print("-" * 20)

        # Check for Syrah variations (case-insensitive and whitespace)
        print(f"Checking Syrah variations in {split}:")
        syrah_variations = set()
        for example in dataset[split]:
            variety = example['variety']
            if variety is not None:
                if 'syrah' in variety.lower(): #case insensitive
                    syrah_variations.add(variety)
        print(syrah_variations)

if __name__ == "__main__":
    check_syrah_distribution()