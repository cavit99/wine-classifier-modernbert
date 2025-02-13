"""
Wine Variety Predictor
Author: Cavit Erginsoy
Year: 2025
License: MIT License
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import json

def run_inference(review_text: str) -> str:
    """
    Perform inference on the given wine review text and return the predicted wine variety
    using ModernBERT, an encoder-only classifier from "spawn99/modernbert-wine-classification".

    Args:
        review_text (str): Wine review text in the format "country [SEP] description".

    Returns:
        str: The predicted wine variety using the model's id2label mapping if available.
    """
    # Define model and tokenizer identifiers
    model_id = "spawn99/modernbert-wine-classification"
    tokenizer_id = "answerdotai/ModernBERT-base"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # The model used here is a ModernBERT encoder-only classifier.
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Tokenize the input text
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

    # Determine prediction and map to label if available
    pred = torch.argmax(logits, dim=-1).item()
    variety = (
        model.config.id2label.get(pred, str(pred))
        if hasattr(model.config, "id2label") and model.config.id2label
        else str(pred)
    )
    
    return variety


def predict_wine_variety(country: str, description: str, output_format: str) -> str:
    """
    Combine the provided country and description, perform inference, and format the output
    based on the selected output format.

    Enforces a maximum character limit of 750 on the description.

    Args:
        country (str): The country of wine origin.
        description (str): The wine review description.
        output_format (str): Either "JSON" to return output as a JSON-formatted string,
                             or "Text" for plain text output.

    Returns:
        str: The predicted wine variety formatted as JSON (if selected) or as plain text.
    """
    if len(description) > 750:
        error_msg = "Description exceeds 750 character limit. Please shorten your input."
        if output_format.lower() == "json":
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return error_msg
    
    # Capitalize input values and format the review text accordingly.
    review_text = f"{country.capitalize()} [SEP] {description.capitalize()}"
    predicted_variety = run_inference(review_text)
    
    if output_format.lower() == "json":
        return json.dumps({"Variety": predicted_variety}, indent=2)
    else:
        return predicted_variety


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict_wine_variety,
        inputs=[
            gr.Textbox(label="Country", placeholder="Enter country of origin..."),
            gr.Textbox(label="Description", placeholder="Enter wine review description..."),
            gr.Radio(choices=["JSON", "Text"], value="JSON", label="Output Format")
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Wine Variety Predictor",
        description=(
            "Predict the wine variety based on the country and wine review.\n\n"
            "This tool uses [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), "
            "an encoder-only classifier, trained on the [wine reviews dataset]"
            "(https://huggingface.co/datasets/spawn99/wine-reviews)\n\n"
            "** Resources:**\n"
            "- Repository: [GitHub:cavit99/wine-classifier-modernbert](https://github.com/cavit99/wine-classifier-modernbert)\n"
            "- Model: [spawn99/modernbert-wine-classification](https://huggingface.co/spawn99/modernbert-wine-classification)\n"
            "- Dataset: [spawn99/wine-reviews](https://huggingface.co/datasets/spawn99/wine-reviews)\n\n"
            "*Cavit Erginsoy, 2025*\n"
        )
    )
    iface.launch()