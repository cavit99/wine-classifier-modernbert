import numpy as np
from datasets import  DatasetDict, load_from_disk, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import torch
from collections import Counter


class WeightedModernBERTTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Convert logits to full precision to aid numerical stability, especially in mixed-precision training.
        logits = outputs.logits.float()
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_and_prepare_data(min_count: int = 50, blend_threshold: int = 150) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """
    Global Filtering Before Splitting (Stratified):
    
    Loads the wine reviews dataset from the Hugging Face Hub, combines all splits,
    creates a "text" column, normalizes the 'variety' column,
    conducts blend normalization, 
    filters rare classes, and then performs a stratified split into train, validation, and test sets.
    
    Additionally, it ensures that each split has at least one sample per class (based on the training set).
    """
    # Load the dataset from the Hugging Face Hub (original splits will be ignored)
    original_dataset = load_dataset("spawn99/wine-reviews")
    print("Loaded dataset from Hugging Face Hub:")
    print({split: type(original_dataset[split]) for split in original_dataset.keys()})
    
    # Combine all splits into one global dataset
    full_dataset = concatenate_datasets([original_dataset[split] for split in original_dataset.keys()])
    
    # 1. Combine 'country' and 'description' into a 'text' column.
    def create_text_column(example: dict) -> dict:
        example['text'] = f"{example['country']} [SEP] {example['description']}"
        return example
    full_dataset = full_dataset.map(create_text_column)
    
    # 2. Initial normalization of the 'variety' column.
    def custom_normalize(example: dict) -> dict:
        # Check if the 'variety' field is None
        if example['variety'] is None:
            example['drop'] = True
            return example

        var = example['variety'].strip()
        var_lower = var.lower()
        
        # Mark rows to drop if the variety is exactly "red blend" or "white blend" or if "sparkling" is present.
        example['drop'] = var_lower in ['red blend', 'white blend'] or 'sparkling' in var_lower
        example['variety'] = var 
        return example
    full_dataset = full_dataset.map(custom_normalize)
    full_dataset = full_dataset.filter(lambda ex: not ex['drop'])
    full_dataset = full_dataset.remove_columns("drop")
    
    # 3. Compute global blend frequency counts.
    blend_counts = {}
    for example in full_dataset:
        var = example['variety']
        if '-' in var:
            blend_counts[var] = blend_counts.get(var, 0) + 1
    
    # 4. Normalize blends based on their global frequency.
    def normalize_blends(example: dict) -> dict:
        var = example['variety']
        if '-' in var:
            if blend_counts.get(var, 0) < blend_threshold:
                left_grape = var.split('-')[0].strip()
                example['variety'] = left_grape + " Blend"
        return example
    full_dataset = full_dataset.map(normalize_blends)
    
    # 5. Global Filtering: Remove rare classes based on overall variety counts.
    global_counts = Counter(full_dataset["variety"])
    def add_global_count(example: dict) -> dict:
        example["variety_counts"] = global_counts.get(example["variety"], 0)
        return example
    full_dataset = full_dataset.map(add_global_count)
    full_dataset = full_dataset.filter(lambda ex: ex['variety_counts'] >= min_count)
    full_dataset = full_dataset.remove_columns("variety_counts")
    
    # 6. Only keep the necessary columns.
    columns_to_keep = ['text', 'variety']
    columns_to_remove = [col for col in full_dataset.column_names if col not in columns_to_keep]
    full_dataset = full_dataset.remove_columns(columns_to_remove)
    
    # 7. Perform stratified splitting into train, validation, and test sets.
    # Attempt splitting multiple times if any split is missing classes.
    max_attempts = 10
    for attempt in range(max_attempts):
        current_seed = 42 + attempt
        # First, split 70% for training and 30% as a temporary split.
        split1 = full_dataset.train_test_split(test_size=0.3, stratify_by_column="variety", seed=current_seed)
        # Then, further split the temporary 30% into 15% validation and 15% test sets.
        temp_split = split1["test"].train_test_split(test_size=0.5, stratify_by_column="variety", seed=current_seed)
        dataset_splits = {
            "train": split1["train"],
            "validation": temp_split["train"],
            "test": temp_split["test"]
        }
        # 8. Create label mappings based solely on the training set.
        unique_varieties = sorted(list(set(dataset_splits['train']['variety'])))
        label2id = {label: i for i, label in enumerate(unique_varieties)}
        id2label = {i: label for label, i in label2id.items()}
        
        if validate_splits(dataset_splits, label2id):
            break
        else:
            print(f"Resplitting attempt {attempt+1} failed. Trying new split...")
    else:
        print("Warning: Could not achieve valid splits after multiple attempts. Proceeding with current splits.")
    
    return dataset_splits, label2id, id2label


def train_model(dataset, label2id, id2label):
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        tokenized_dataset = load_from_disk("tokenized_wine_dataset")
        print("Tokenized dataset loaded from disk.")
    except Exception as e:
        print("Tokenized dataset not found. Tokenizing now...")
        def tokenize_function(examples):
            tokens = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )
            tokens["labels"] = [label2id[variety] for variety in examples["variety"]]
            return tokens
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.save_to_disk("tokenized_wine_dataset")

    train_labels = tokenized_dataset['train']['labels']
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

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
        hub_model_id="spawn99/modernbert-wine-classification",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
    )

    trainer = WeightedModernBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        class_weights=class_weights,
        compute_metrics=lambda eval_pred: compute_metrics(training_args, eval_pred),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.push_to_hub()
    return model, tokenizer, tokenized_dataset, training_args


def compute_metrics(training_args, eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    f1_val = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1_val}


def evaluate_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokenized_dataset: DatasetDict,
    label2id: dict[str, int],
    id2label: dict[int, str],
    training_args: TrainingArguments
):
    # Disable any reporting during evaluation by setting the report_to list to empty.
    training_args.report_to = []

    # Initialize the trainer for evaluation using the updated training_args.
    trainer = WeightedModernBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=tokenized_dataset["test"],
        class_weights=None,  
        compute_metrics=lambda eval_pred: compute_metrics(training_args, eval_pred),
    )

    # Evaluate the model for metrics like loss (this returns a dict).
    eval_metrics = trainer.evaluate()
    print(f"Evaluation Metrics: {eval_metrics}")

    # Use trainer.predict() to obtain predictions and labels.
    prediction_output = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(prediction_output.predictions, axis=1)
    labels = prediction_output.label_ids

    # Convert numeric labels to string labels
    predictions = [id2label[p] for p in predictions]
    labels = [id2label[l] for l in labels]

    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate per-class F1 scores and other metrics
    report = classification_report(
        labels, predictions, target_names=list(id2label.values()), output_dict=True
    )

    # Print F1 scores for tail classes (varieties with < 100 samples in the training set)
    print("\nPer-Class F1 Scores (for varieties with < 100 samples in the training set):")
    train_counts = tokenized_dataset["train"].to_pandas()["labels"].value_counts()
    for variety, metrics in report.items():
        if variety in id2label.values():
            variety_id = label2id[variety]
            if variety_id in train_counts.index and train_counts[variety_id] < 100:
                print(f"  {variety}: F1 = {metrics['f1-score']:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(id2label.values()),
        yticklabels=list(id2label.values()),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def validate_splits(dataset_splits, label2id):
    """
    Validate that each split has at least one sample per class.
    Return True if all splits contain at least one sample for every class,
    otherwise return False and print warnings.
    """
    valid = True
    for split_name, dataset in dataset_splits.items():
        counts = Counter(dataset["variety"])
        missing = [cls for cls in label2id if cls not in counts or counts[cls] == 0]
        if missing:
            print(f"Warning: In the '{split_name}' split, the following classes are missing: {missing}")
            valid = False
    return valid


def main():
    # Load and prepare data, filtering rare varieties
    dataset, label2id, id2label = load_and_prepare_data(min_count=50)
    validate_splits(dataset, label2id)

    print("Training model...")
    model, tokenizer, tokenized_dataset, training_args = train_model(dataset, label2id, id2label)
    evaluate_model(model, tokenizer, tokenized_dataset, label2id, id2label, training_args)


if __name__ == "__main__":
    main()