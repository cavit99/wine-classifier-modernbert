import numpy as np
from datasets import  DatasetDict, load_from_disk, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import torch


class WeightedModernBERTTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        if self.lr_scheduler is None:  # Check if scheduler already exists
            self.lr_scheduler = WSDLearningRateScheduler(
                optimizer=optimizer,
                warmup_ratio=self.args.warmup_ratio,
                decay_ratio=0.1,
                min_lr_ratio=0.1,
                total_steps=num_training_steps,
            )
        return self.lr_scheduler


class WSDLearningRateScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_ratio, decay_ratio, min_lr_ratio, total_steps, last_epoch=-1):
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = total_steps
        self.peak_lr = None
        self.min_lr = None

        self.warmup_steps = int(self.warmup_ratio * self.total_steps)
        self.decay_steps = int(self.decay_ratio * self.total_steps)
        self.stable_steps = self.total_steps - self.warmup_steps - self.decay_steps

        if self.stable_steps < 0:
            raise ValueError("Total steps are too small for the given warmup and decay ratios.")

        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if self.peak_lr is None:  # Initialize peak_lr and min_lr here
            self.peak_lr = self.optimizer.defaults['lr']
            self.min_lr = self.peak_lr * self.min_lr_ratio
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        elif current_step < self.warmup_steps + self.stable_steps:
            return 1.0
        elif current_step < self.warmup_steps + self.stable_steps + self.decay_steps:
            decay_step = current_step - self.warmup_steps - self.stable_steps
            decay_step = min(decay_step, self.decay_steps - 1)  # Clamp, as in PPLX
            return (self.min_lr * self.peak_lr) / (decay_step / self.decay_steps * (self.peak_lr - self.min_lr) + self.min_lr)
        else:
            return self.min_lr / self.peak_lr



def load_and_prepare_data(min_count: int = 50) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """
    Loads the wine reviews dataset directly from the Hugging Face Hub,
    creates a "text" column (combining country and description), filters
    out rare wine varieties, and builds label mappings.
    """
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset("spawn99/wine-reviews")
    print("Loaded dataset from Hugging Face Hub:")
    print({split: type(dataset[split]) for split in dataset.keys()})

    # 1. Combine 'country' and 'description' into a 'text' column.
    def create_text_column(example: dict) -> dict:
        example['text'] = f"{example['country']} [SEP] {example['description']}"
        return example
    dataset = dataset.map(create_text_column)

    # 2. Filter out rare varieties *before* splitting if needed.
    #    First, collect variety counts from the entire dataset
    def count_varieties(examples: dict) -> dict:
        counts = {}
        for variety in examples['variety']:
            counts[variety] = counts.get(variety, 0) + 1
        return {'variety_counts': [counts.get(v,0) for v in examples['variety']]}

    dataset = dataset.map(count_varieties, batched=True)
    dataset = dataset.filter(lambda example: example['variety_counts'] >= min_count)
    dataset = dataset.remove_columns('variety_counts')

    # 3. Create label mappings based on the *filtered* training data.
    unique_varieties = sorted(list(set(dataset['train']['variety'])))
    label2id = {label: i for i, label in enumerate(unique_varieties)}
    id2label = {i: label for label, i in label2id.items()}

    # 4. Select only necessary columns
    dataset = dataset.remove_columns([col for col in dataset['train'].column_names if col not in ['text', 'variety']])

    return dataset, label2id, id2label

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
            tokens["label"] = [label2id[variety] for variety in examples["variety"]]
            return tokens
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.save_to_disk("tokenized_wine_dataset")

    train_labels = tokenized_dataset['train']['label']
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
        learning_rate=1e-4,
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
        warmup_ratio=0.08,
        lr_scheduler_type="linear",
    )

    # Do not override the Trainer's default optimizer and scheduler.
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

def evaluate_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, tokenized_dataset: DatasetDict, label2id: dict[str, int], id2label: dict[int, str], training_args: TrainingArguments):
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

    # Evaluate the model.
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Extract predictions and labels from the evaluation results
    predictions = np.argmax(eval_results.predictions, axis=1)
    labels = eval_results.label_ids

    # Convert numeric labels to string labels
    predictions = [id2label[p] for p in predictions]
    labels = [id2label[l] for l in labels]

    # Calculate accuracy
    accuracy = sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate per-class F1 scores and other metrics
    report = classification_report(labels, predictions, target_names=list(id2label.values()), output_dict=True)

    # Print F1 scores for tail classes (varieties with < 100 samples in the *training* set):
    print("\nPer-Class F1 Scores (for varieties with < 100 samples in the *training* set):")
    train_counts = tokenized_dataset['train'].to_pandas()['label'].value_counts()
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
        fmt='d',
        cmap='Blues',
        xticklabels=list(id2label.values()),
        yticklabels=list(id2label.values())
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():  # No parameters needed
    # Load and prepare data, filtering rare varieties
    dataset, label2id, id2label = load_and_prepare_data(min_count=50)

    print("Training model...")
    model, tokenizer, tokenized_dataset, training_args = train_model(dataset, label2id, id2label)  
    evaluate_model(model, tokenizer, tokenized_dataset, label2id, id2label, training_args)  

if __name__ == "__main__":
    main()