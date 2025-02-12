from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path to your best model checkpoint (e.g., where your Trainer has saved the best model)
best_model_checkpoint = "./path/to/your/best_checkpoint"
repo_name = "your-username/your-best-model"  # Replace with your actual Hugging Face repo name

def push_model_to_hub():
    # Load the model and tokenizer from the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(best_model_checkpoint)

    print("Pushing model to the Hub...")
    model.push_to_hub(repo_name, commit_message="Pushing best model checkpoint")
    tokenizer.push_to_hub(repo_name)
    print("Upload completed.")

if __name__ == "__main__":
    push_model_to_hub() 