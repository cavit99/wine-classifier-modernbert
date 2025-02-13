import math
from datasets import load_from_disk
from transformers import TrainingArguments

def main():
    try:
        # Load the tokenized dataset from disk
        tokenized_dataset = load_from_disk("tokenized_wine_dataset")
        print("Tokenized dataset loaded from disk.")
    except Exception as e:
        print("Tokenized dataset could not be loaded. Please run the training script first to tokenize the dataset.")
        return

    # Create a TrainingArguments object with the current training configuration.
    # These should match the settings in train.py.
    training_args = TrainingArguments(
        output_dir="modernbert-winevariety",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=5e-5,
        num_train_epochs=5,
        logging_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="modernbert-wine-classification",
        metric_for_best_model="f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        bf16=True,
        push_to_hub=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
    )

    # Calculate steps based solely on the training set.
    train_dataset = tokenized_dataset["train"]
    num_train_examples = len(train_dataset)
    batch_size = training_args.per_device_train_batch_size  # assuming one device in this calculation
    steps_per_epoch = math.ceil(num_train_examples / batch_size)
    total_training_steps = steps_per_epoch * training_args.num_train_epochs

    print("\n--- Training Step Calculations ---")
    print(f"Number of training examples: {num_train_examples}")
    print(f"Per-device train batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Total training steps: {total_training_steps}")
    print("\nAdditional configuration values:")
    print(f"Logging steps: {training_args.logging_steps}")
    print(f"Evaluation steps: {training_args.eval_steps}")
    print(f"Save steps: {training_args.save_steps}")

if __name__ == "__main__":
    main() 