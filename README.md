# Cell Classification

This project performs classification of cell images into three classes (5nMTG, Control, NonCancerOral) and evaluates the model performance using various metrics.

## Dataset Structure

The dataset is organized as follows:

```
Data/
├── oral cancer 0521-0618_tag300_Test/
│   ├── 5nMTG/
│   ├── Control/
│   └── NonCancerOral/
└── oral cancer 0521-0618_tag300_Val/
    ├── 5nMTG/
    ├── Control/
    └── NonCancerOral/
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the classification script:

```bash
python cell_classification.py
```

## Output

The script will:

1. Train a ResNet50-based model on the validation set
2. Evaluate the model on the test set
3. Calculate and display the following metrics:
   - Precision
   - Recall
   - Specificity
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
   - AUC-ROC with AUC Score
4. Generate visualizations:
   - ROC curves for each class
   - Confusion matrix
   - Training history (accuracy and loss)

## Visualizations

The script generates the following visualization files:

- `roc_curve.png`: ROC curves for each class with AUC scores
- `confusion_matrix.png`: Confusion matrix showing prediction results
- `training_history.png`: Training and validation accuracy/loss over epochs
