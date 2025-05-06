import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths and parameters
results_dir = 'improved_cell_classification_results'
ensemble_dir = 'ensemble_cell_classification_results'
os.makedirs(ensemble_dir, exist_ok=True)

test_dir = 'Data/oral cancer 0521-0618_tag300_Test'
batch_size = 4
num_classes = 3
class_names = ['5nMTG', 'Control', 'NonCancerOral']

# Define model configurations
model_configs = {
    'ResNet152': {
        'preprocess_func': resnet_preprocess,
        'target_size': (224, 224)
    },
    'ResNet101V2': {
        'preprocess_func': resnet_v2_preprocess,
        'target_size': (224, 224)
    },
    'NASNetMobile': {
        'preprocess_func': nasnet_preprocess,
        'target_size': (224, 224)
    },
    'Xception': {
        'preprocess_func': xception_preprocess,
        'target_size': (299, 299)
    }
}

# Function to create test data generator for each model
def create_test_generator(model_name):
    config = model_configs[model_name]
    test_datagen = ImageDataGenerator(
        preprocessing_function=config['preprocess_func']
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=config['target_size'],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

# Function to calculate metrics (same as in the original script)
def calculate_metrics(y_true, y_pred, y_pred_prob):
    # Convert to one-hot encoding for ROC curve
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate F1 score directly for each class to handle zero division properly
    f1 = []
    for i in range(num_classes):
        if precision[i] + recall[i] > 0:
            f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
        else:
            f1.append(0.0)
    f1 = np.array(f1)
    
    # Calculate macro averages
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate macro F1 directly from macro precision and recall
    if precision_macro + recall_macro > 0:
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    else:
        f1_macro = 0.0
    
    # Double-check with sklearn's implementation
    f1_macro_check = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # If there's a significant difference, use sklearn's implementation
    if abs(f1_macro - f1_macro_check) > 0.01:
        f1_macro = f1_macro_check
    
    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate specificity for each class
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(num_classes):
        true_neg = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_pos = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0)
    
    # Calculate macro average for specificity
    specificity_macro = np.mean(specificity)
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'specificity_macro': specificity_macro,
        'f1': f1,
        'mcc': mcc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

# Function to print metrics (same as in the original script)
def print_metrics(metrics, model_name, accuracy):
    # Verify F1 scores are calculated correctly
    for i in range(num_classes):
        p = metrics['precision'][i]
        r = metrics['recall'][i]
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            if abs(expected_f1 - metrics['f1'][i]) > 0.01:
                metrics['f1'][i] = expected_f1
    
    # Verify macro F1 score
    p_macro = metrics['precision_macro']
    r_macro = metrics['recall_macro']
    if p_macro + r_macro > 0:
        expected_f1_macro = 2 * p_macro * r_macro / (p_macro + r_macro)
        if abs(expected_f1_macro - metrics['f1_macro']) > 0.01:
            metrics['f1_macro'] = expected_f1_macro
    
    print(f"\n===== {model_name} Classification Metrics =====")
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\n1) Precision:")
    for i, class_name in enumerate(class_names):
        print(f"   - {class_name}: {metrics['precision'][i]:.4f}")
    print(f"   - Macro Average: {metrics['precision_macro']:.4f}")
    
    print("\n2) Recall:")
    for i, class_name in enumerate(class_names):
        print(f"   - {class_name}: {metrics['recall'][i]:.4f}")
    print(f"   - Macro Average: {metrics['recall_macro']:.4f}")
    
    print("\n3) Specificity:")
    for i, class_name in enumerate(class_names):
        print(f"   - {class_name}: {metrics['specificity'][i]:.4f}")
    print(f"   - Macro Average: {metrics['specificity_macro']:.4f}")
    
    print("\n4) F1 Score:")
    for i, class_name in enumerate(class_names):
        print(f"   - {class_name}: {metrics['f1'][i]:.4f}")
    print(f"   - Macro Average: {metrics['f1_macro']:.4f}")
    
    print(f"\n5) MCC Score: {metrics['mcc']:.4f}")

# Function to plot ROC curve (same as in the original script)
def plot_roc_curve(metrics, model_name):
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, color, class_name in zip(range(num_classes), ['blue', 'red', 'green'], class_names):
        plt.plot(
            metrics['fpr'][i],
            metrics['tpr'][i],
            color=color,
            lw=2,
            label=f'{class_name} (AUC = {metrics["roc_auc"][i]:.2f})'
        )
    
    # Plot micro-average ROC curve
    plt.plot(
        metrics['fpr']["micro"],
        metrics['tpr']["micro"],
        color='deeppink',
        linestyle=':',
        linewidth=4,
        label=f'Micro-average (AUC = {metrics["roc_auc"]["micro"]:.2f})'
    )
    
    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(ensemble_dir, f'roc_curve_{model_name}.png'), dpi=300)
    plt.close()

# Function to save results to file
def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ENSEMBLE CELL CLASSIFICATION MODEL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        metrics = results['metrics']
        accuracy = results['accuracy']
        
        f.write(f"MODEL: Ensemble of ResNet152, ResNet101V2, NASNetMobile, and Xception\n")
        f.write("-" * 60 + "\n\n")
        
        # Write performance metrics
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        
        f.write("1) Precision:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"   - {class_name}: {metrics['precision'][i]:.4f}\n")
        f.write(f"   - Macro Average: {metrics['precision_macro']:.4f}\n\n")
        
        f.write("2) Recall:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"   - {class_name}: {metrics['recall'][i]:.4f}\n")
        f.write(f"   - Macro Average: {metrics['recall_macro']:.4f}\n\n")
        
        f.write("3) Specificity:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"   - {class_name}: {metrics['specificity'][i]:.4f}\n")
        f.write(f"   - Macro Average: {metrics['specificity_macro']:.4f}\n\n")
        
        f.write("4) F1 Score:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"   - {class_name}: {metrics['f1'][i]:.4f}\n")
        f.write(f"   - Macro Average: {metrics['f1_macro']:.4f}\n\n")
        
        f.write(f"5) MCC Score: {metrics['mcc']:.4f}\n\n")
        
        f.write(f"6) AUC Score (micro-average): {metrics['roc_auc']['micro']:.4f}\n\n")

# Main execution
print("\n\n" + "="*50)
print("ENSEMBLE CELL CLASSIFICATION")
print("="*50)

# Load the best models
models = {}
for model_name in model_configs.keys():
    try:
        model_path = os.path.join(results_dir, f'best_{model_name.lower()}_model.keras')
        models[model_name] = load_model(model_path)
        print(f"Loaded {model_name} model from {model_path}")
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")

# Get predictions from each model
all_predictions = {}
y_true = None

for model_name, model in models.items():
    test_generator = create_test_generator(model_name)
    
    # Get ground truth labels (same for all models)
    if y_true is None:
        y_true = test_generator.classes
    
    # Get predictions
    y_pred_prob = model.predict(test_generator)
    all_predictions[model_name] = y_pred_prob

# Create ensemble predictions using weighted averaging
# Weights based on individual model performance (can be adjusted)
ensemble_weights = {
    'ResNet152': 0.35,
    'ResNet101V2': 0.30,
    'NASNetMobile': 0.20,
    'Xception': 0.15
}

# Normalize weights to sum to 1
total_weight = sum(ensemble_weights.values())
for model_name in ensemble_weights:
    ensemble_weights[model_name] /= total_weight

# Compute weighted average predictions
ensemble_pred_prob = np.zeros_like(next(iter(all_predictions.values())))
for model_name, pred_prob in all_predictions.items():
    if model_name in ensemble_weights:
        ensemble_pred_prob += pred_prob * ensemble_weights[model_name]

# Get class predictions
ensemble_pred = np.argmax(ensemble_pred_prob, axis=1)

# Calculate accuracy
ensemble_accuracy = np.mean(ensemble_pred == y_true)
print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")

# Calculate all metrics
ensemble_metrics = calculate_metrics(y_true, ensemble_pred, ensemble_pred_prob)

# Store results
ensemble_results = {
    'metrics': ensemble_metrics,
    'accuracy': ensemble_accuracy
}

# Print metrics
print_metrics(ensemble_metrics, "Ensemble", ensemble_accuracy)

# Plot ROC curve
plot_roc_curve(ensemble_metrics, "Ensemble")

# Plot confusion matrix
cm = confusion_matrix(y_true, ensemble_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Ensemble Model')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add text annotations to the confusion matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(ensemble_dir, 'confusion_matrix_ensemble.png'), dpi=300)
plt.close()

# Save results to file
save_results_to_file(ensemble_results, os.path.join(ensemble_dir, "ensemble_results.txt"))

print(f"\nEnsemble results saved to the '{ensemble_dir}' folder") 