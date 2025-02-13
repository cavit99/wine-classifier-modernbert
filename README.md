```markdown:README.md
# Wine Variety Classification with ModernBERT

This project classifies wine varieties based on country and description using a fine-tuned ModernBERT model. The system handles class imbalance through weighted loss and employs careful dataset stratification.

## Features

- 🍷 Trained on combined country + description
- ⚖️ Class imbalance handling with weighted loss
- 🧪 Stratified dataset splitting with rare class filtering
- 📊 Evaluation with per-class F1 scores and confusion matrix
- 🤗 Hugging Face Hub integration for dataset and model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cavit99/wine-classification.git
cd wine-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up WandB for experiment tracking (optional but recommended):
```bash
wandb login
```

## Dataset

The wine reviews dataset is hosted on Hugging Face Hub:
```python
from datasets import load_dataset
dataset = load_dataset("spawn99/wine-reviews")
```

### Dataset Structure
- **text**: Combined country and description (separated by [SEP])
- **variety**: Wine variety label (normalized and filtered)

## Usage

### 1. Preprocessing
```bash
python preprocess_dataset.py
```

### 2. Training
```bash
python train.py
```

### 3. Evaluation
Evaluation metrics and confusion matrix are automatically generated after training.

## Model Details

**Architecture**: `answerdotai/ModernBERT-base`  
**Training**:
- Batch size: 128
- Learning rate: 5e-5 (cosine scheduler)
- Early stopping: 3 epochs patience
- Class-weighted cross-entropy loss

**Performance**:
- Weighted F1 score
- Accuracy
- Per-class metrics for rare varieties

## Key Implementation Details

- **Class Normalization**:
  - Blend varieties normalized to "[Grape] Blend"
  - Rare varieties filtered (<50 samples)
  
- **Stratified Splitting**:
  - 70% train / 10% validation / 20% test
  - Ensures all splits contain all classes

## Requirements

See [requirements.txt](requirements.txt) for full dependency list.

## License

MIT License - see [LICENSE](LICENSE) for details.
```