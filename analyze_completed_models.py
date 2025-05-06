import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
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
results_dir = 'enhanced_cell_classification_results'
analysis_dir = 'model_analysis_results'
os.makedirs(analysis_dir, exist_ok=True)

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

# Function to calculate metrics
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
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

# Function to print metrics
def print_metrics(metrics, model_name, accuracy):
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
    
    print(f"\n6) AUC Score (micro-average): {metrics['roc_auc']['micro']:.4f}")

# Function to plot ROC curve
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
    plt.savefig(os.path.join(analysis_dir, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
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
    plt.savefig(os.path.join(analysis_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
    plt.close()

# Function to create a bar chart comparing model performance
def plot_model_comparison(all_results):
    # Sort models by accuracy
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    model_names = [model_name for model_name, _ in sorted_models]
    
    # Extract metrics
    accuracies = [results['accuracy'] for _, results in sorted_models]
    f1_scores = [results['metrics']['f1_macro'] for _, results in sorted_models]
    mcc_scores = [results['metrics']['mcc'] for _, results in sorted_models]
    auc_scores = [results['metrics']['roc_auc']['micro'] for _, results in sorted_models]
    
    # Create bar chart
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width*1.5, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x - width/2, f1_scores, width, label='F1 Score')
    rects3 = ax.bar(x + width/2, mcc_scores, width, label='MCC')
    rects4 = ax.bar(x + width*1.5, auc_scores, width, label='AUC')
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'model_comparison.png'), dpi=300)
    plt.close()

# Function to create a radar chart for each model
def plot_radar_chart(all_results):
    # Define metrics to include in radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'MCC', 'AUC']
    
    # Create a figure for all models
    fig = plt.figure(figsize=(15, 10))
    
    # Sort models by accuracy
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Create a radar chart for each model
    for i, (model_name, results) in enumerate(sorted_models):
        # Extract metrics
        accuracy = results['accuracy']
        precision = results['metrics']['precision_macro']
        recall = results['metrics']['recall_macro']
        specificity = results['metrics']['specificity_macro']
        f1 = results['metrics']['f1_macro']
        mcc = results['metrics']['mcc']
        auc_score = results['metrics']['roc_auc']['micro']
        
        # Combine metrics
        values = [accuracy, precision, recall, specificity, f1, mcc, auc_score]
        
        # Create radar chart
        ax = fig.add_subplot(2, 2, i+1, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Values for each metric
        values += values[:1]  # Close the loop
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Set title
        ax.set_title(model_name, size=15, y=1.1)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'model_radar_charts.png'), dpi=300)
    plt.close()

# Function to save detailed results to file
def save_detailed_results(all_results, filename):
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED MODEL ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort models by accuracy
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, results in sorted_models:
            metrics = results['metrics']
            accuracy = results['accuracy']
            
            f.write(f"MODEL: {model_name}\n")
            f.write("-" * 50 + "\n\n")
            
            # Write performance metrics
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            
            f.write("1) Precision:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"   - {class_name}: {metrics['precision'][i]:.4f}\n")
            f.write(f"   - Macro Average: {metrics['precision_macro']:.4f}\n\n")
            
            f.write("2) Recall (Sensitivity):\n")
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
            
            f.write("7) Confusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(f"    {' ' * 15} Predicted\n")
            f.write(f"    {' ' * 15} {class_names[0]:<10} {class_names[1]:<10} {class_names[2]:<10}\n")
            f.write(f"    Actual\n")
            for i, class_name in enumerate(class_names):
                f.write(f"    {class_name:<10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY COMPARISON\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<10} {'F1 Score':<10} {'MCC':<10} {'AUC':<10}\n")
        f.write("-" * 90 + "\n")
        
        for model_name, results in sorted_models:
            metrics = results['metrics']
            accuracy = results['accuracy']
            f.write(f"{model_name:<15} {accuracy:.4f}      {metrics['precision_macro']:.4f}     {metrics['recall_macro']:.4f}     {metrics['specificity_macro']:.4f}     {metrics['f1_macro']:.4f}      {metrics['mcc']:.4f}     {metrics['roc_auc']['micro']:.4f}\n")
        
        f.write("\n\nAnalysis completed on: " + time.strftime("%Y-%m-%d %H:%M:%S"))

# Main execution
print("\n\n" + "="*50)
print("ANALYZING COMPLETED MODELS")
print("="*50)

# Load the trained models
all_results = {}

for model_name in model_configs.keys():
    try:
        # Create test generator for this model
        test_generator = create_test_generator(model_name)
        
        # Load the model
        model_path = os.path.join(results_dir, f'best_{model_name.lower()}_model.keras')
        if os.path.exists(model_path):
            print(f"\nAnalyzing {model_name}...")
            model = load_model(model_path)
            
            # Evaluate the model
            test_loss, test_acc = model.evaluate(test_generator, verbose=0)
            print(f'Test accuracy: {test_acc:.4f}')
            
            # Get predictions
            test_generator.reset()
            y_pred_prob = model.predict(test_generator, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = test_generator.classes
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred, y_pred_prob)
            
            # Store results
            all_results[model_name] = {
                'metrics': metrics,
                'accuracy': test_acc
            }
            
            # Print metrics
            print_metrics(metrics, model_name, test_acc)
            
            # Plot ROC curve
            plot_roc_curve(metrics, model_name)
            
            # Plot confusion matrix
            plot_confusion_matrix(metrics['confusion_matrix'], model_name)
            
            # Clear memory
            tf.keras.backend.clear_session()
        else:
            print(f"Model file for {model_name} not found at {model_path}")
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")

# Create comparison visualizations
if len(all_results) > 0:
    print("\nCreating comparison visualizations...")
    plot_model_comparison(all_results)
    plot_radar_chart(all_results)
    
    # Save detailed results
    save_detailed_results(all_results, os.path.join(analysis_dir, "detailed_model_analysis.txt"))
    print(f"\nAnalysis results saved to {analysis_dir}")
else:
    print("\nNo models were successfully analyzed.")

print("\nAnalysis complete!") 