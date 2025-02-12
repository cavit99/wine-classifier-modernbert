import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import torch
import os

class WeightedModernBERTTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_and_prepare_data(min_count=50):
    processed_data_path = "processed_wine_data.pkl"

    if os.path.exists(processed_data_path):
        print("Loading preprocessed data...")
        df = pd.read_pickle(processed_data_path)
    else:
        # Load data
        df1 = pd.read_csv("dataset/winemag-data_first150k.csv")
        df2 = pd.read_csv("dataset/winemag-data-130k-v2.csv")

        # Select relevant columns and handle missing values
        df1 = df1[["country", "description", "variety"]]
        df2 = df2[["country", "description", "variety"]]
        df1 = df1.dropna(subset=["country", "description", "variety"])
        df2 = df2.dropna(subset=["country", "description", "variety"])

        # Combine dataframes
        df = pd.concat([df1, df2], ignore_index=True)

        # Combine country and description
        df["text"] = df["country"] + " [SEP] " + df["description"]
        
        # Filter out rare varieties
        variety_counts = df["variety"].value_counts()
        varieties_to_keep = variety_counts[variety_counts >= min_count].index
        df = df[df["variety"].isin(varieties_to_keep)]
        print(f"Number of examples after filtering: {len(df)}")

        # Create train/test split
        train_fraction = 0.8
        random_mask = np.random.RandomState(42).rand(len(df)) < train_fraction
        df["train_test_split"] = np.where(random_mask, "train", "test")

        df.to_pickle(processed_data_path)

    # Create label mappings
    unique_varieties = df["variety"].unique()
    label2id = {label: i for i, label in enumerate(sorted(unique_varieties))}
    id2label = {i: label for label, i in label2id.items()}
    df['label'] = df['variety'].map(label2id)
    print(f"Number of unique varieties: {len(unique_varieties)}")

    # Convert to HuggingFace Dataset
    dataset = DatasetDict({
        'train': Dataset.from_pandas(df[df['train_test_split'] == 'train']),
        'test': Dataset.from_pandas(df[df['train_test_split'] == 'test'])
    })

    print(f"Training set size: {len(dataset['train'])}")
    return dataset, label2id, id2label

def train_model(dataset, label2id, id2label):
    # Initialize tokenizer and model
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # --- LOAD DATASET IF IT EXISTS ---
    try:
        tokenized_dataset = load_from_disk("tokenized_wine_dataset")
        print("Tokenized dataset loaded from disk.")
    except:
        print("Tokenized dataset not found. Tokenizing...")
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=256
            )

        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.save_to_disk("tokenized_wine_dataset")  # Save it!

    # Calculate class weights
    train_labels = tokenized_dataset['train']['label']
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="modernbert-variety",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        num_train_epochs=4,
        logging_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        skip_memory_metrics=True,
        bf16=True,
    )

    # Initialize trainer
    trainer = WeightedModernBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        class_weights=class_weights,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    return model, tokenizer, tokenized_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"f1": f1}

def evaluate_model(model, tokenizer, tokenized_dataset, label2id, id2label):
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    predictions = []
    labels = []
    
    for item in tqdm(tokenized_dataset['test']):
        pred = classifier(item['text'])[0]
        predictions.append(int(pred['label'].split('_')[-1]))
        labels.append(item['label'])
    
    # Calculate accuracy
    accuracy = sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Calculate per-class F1 scores and other metrics
    report = classification_report(labels, predictions, target_names=list(id2label.values()), output_dict=True)

    # Print F1 scores for tail classes (varieties with < 100 samples in the *training* set):
    print("\nPer-Class F1 Scores (for varieties with < 100 samples in the *training* set):")
    for variety, metrics in report.items():
        if variety in id2label.values(): 
            variety_id = label2id[variety]
            train_counts = tokenized_dataset['train'].to_pandas()['label'].value_counts()
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

def main():
    # Load and prepare data, filtering rare varieties
    dataset, label2id, id2label = load_and_prepare_data(min_count=50)

    print("Training model...")
    model, tokenizer, tokenized_dataset = train_model(dataset, label2id, id2label)
    evaluate_model(model, tokenizer, tokenized_dataset, label2id, id2label)

if __name__ == "__main__":
    main() 