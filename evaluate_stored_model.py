#!/usr/bin/env python3
"""
A script to load a stored model checkpoint and run evaluation on the test split
of a tokenized dataset.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# If desired, you can define a custom Trainer subclass.
# In this example we include the custom compute_loss (for compatibility with the training script),
# although for evaluation with class_weights=None this is not strictly necessary.
class WeightedModernBERTTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # In evaluation, the labels are required but the weights are not
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if (self.class_weights is not None) else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    f1_val = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1_val}

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on a stored model checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the stored model checkpoint directory (e.g., 'checkpoint-1930')."
    )
    parser.add_argument(
        "--tokenized_dataset_dir",
        type=str,
        default="tokenized_wine_dataset",
        help="Path where the tokenized dataset is saved on disk."
    )
    args = parser.parse_args()

    # Load the tokenized dataset from disk.
    print("Loading tokenized dataset from:", args.tokenized_dataset_dir)
    tokenized_dataset = load_from_disk(args.tokenized_dataset_dir)

    if "test" not in tokenized_dataset:
        raise ValueError("Tokenized dataset must have a 'test' split for evaluation.")

    # Load the model and tokenizer from the checkpoint.
    print("Loading model from checkpoint:", args.checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # Create basic training arguments for evaluation.
    training_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=192,
        report_to=[]  # Disable any integrated logging (such as WandB)
    )

    # Instantiate the trainer.
    trainer = WeightedModernBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=tokenized_dataset["test"],
        class_weights=None,  # Not needed at evaluation time
        compute_metrics=compute_metrics
    )

    # Run evaluation.
    print("Running evaluation...")
    eval_metrics = trainer.evaluate()
    print("Evaluation Metrics:", eval_metrics)

    # Predict labels on the test set.
    prediction_output = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(prediction_output.predictions, axis=1)
    labels = prediction_output.label_ids

    # If the model configuration contains label mappings, use them.
    if hasattr(model.config, "id2label") and model.config.id2label is not None:
        # Convert keys to integers if needed.
        try:
            id2label = {int(k): v for k, v in model.config.id2label.items()}
        except Exception:
            id2label = model.config.id2label

        predictions_str = [id2label.get(int(p), str(p)) for p in predictions]
        labels_str = [id2label.get(int(l), str(l)) for l in labels]
        print("\nClassification Report:")
        print(classification_report(labels_str, predictions_str))
    else:
        print("\nClassification Report (numeric labels):")
        print(classification_report(labels, predictions))

    # Generate and display (or save) a confusion matrix.
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 7))
    if hasattr(model.config, "id2label") and model.config.id2label is not None:
        sorted_keys = sorted(id2label.keys())
        label_names = [id2label[k] for k in sorted_keys]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # If no display is available (for example on a server), save the figure.
    if os.environ.get("DISPLAY", "") == "":
        plt.savefig("confusion_matrix.png")
        print("Confusion matrix saved to 'confusion_matrix.png'.")
    else:
        plt.show()

if __name__ == "__main__":
    main() 