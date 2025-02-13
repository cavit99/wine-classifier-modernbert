import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_inference(review_text: str) -> str:
    """
    Run inference on the given wine review text and return the predicted wine variety.

    Parameters:
        review_text (str): Wine review text formatted as "country [SEP] description"

    Returns:
        str: Predicted wine variety using the model's id2label mapping.
    """
    # Use the model you pushed to the Hugging Face Hub.
    # If you trained with `push_to_hub` using hub_model_id="spawn99/modernbert-wine-classification",
    # then this is the model to load.
    model_id = "spawn99/modernbert-wine-classification"
    # Use the tokenizer from answerdotai/ModernBERT-base
    tokenizer_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # Tokenize the input text.
    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()

    # Use the id2label mapping provided in the model config if available.
    if hasattr(model.config, "id2label") and model.config.id2label:
        variety = model.config.id2label.get(pred, str(pred))
    else:
        variety = str(pred)
    
    return variety

def main():
    # Default review text following the same format used in training.
    default_review = "Italy [SEP] Tar and roses collide with grip – this collective-made wine marries wild strawberry acidity to tannins like suede gloves."
    
    # Optionally allow for command line input:
    # e.g., python inference.py "Italy [SEP] This wine offers fresh aromas of cherry and plum with a smooth finish."
    review_text = default_review if len(sys.argv) < 2 else sys.argv[1]
    
    predicted_variety = run_inference(review_text)
    result = {"Variety": predicted_variety}
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main() 