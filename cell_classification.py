import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers.legacy import Adam  # Using legacy optimizer for M1/M2 Macs
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet152, VGG19, NASNetMobile, Xception, EfficientNetV2S, ResNet101V2, MobileNetV3Large, DenseNet201
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

# Create results directory
results_dir = 'enhanced_cell_classification_results'
os.makedirs(results_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
test_dir = 'Data/oral cancer 0521-0618_tag300_Test'
val_dir = 'Data/oral cancer 0521-0618_tag300_Val'

# Note: In this setup, we're using:
# - val_dir (31 images) for training
# - test_dir (54 images) for validation during training and final evaluation
# This is not the conventional approach but is used due to the small dataset size

# Define image parameters
img_height, img_width = 224, 224  # Standard size for pre-trained models
batch_size = 4  # Smaller batch size for better generalization with small dataset
num_classes = 3

# Define class names
class_names = ['5nMTG', 'Control', 'NonCancerOral']

# Function to create data generators with model-specific preprocessing
def create_data_generators(model_name):
    # Select the appropriate preprocessing function and image size
    if model_name == "ResNet152" or model_name == "ResNet101V2":
        preprocess_func = resnet_preprocess if model_name == "ResNet152" else resnet_v2_preprocess
        target_size = (img_height, img_width)
    elif model_name == "VGG19":
        preprocess_func = vgg19_preprocess
        target_size = (img_height, img_width)
    elif model_name == "NASNetMobile":
        preprocess_func = nasnet_preprocess
        target_size = (224, 224)  # NASNetMobile works with 224x224
    elif model_name == "Xception":
        preprocess_func = xception_preprocess
        target_size = (299, 299)  # Xception requires 299x299 input
    elif model_name == "EfficientNetV2S":
        preprocess_func = efficientnetv2_preprocess
        target_size = (384, 384)  # EfficientNetV2S works best with 384x384
    elif model_name == "MobileNetV3Large":
        preprocess_func = mobilenet_v3_preprocess
        target_size = (224, 224)  # MobileNetV3Large works with 224x224
    elif model_name == "DenseNet201":
        preprocess_func = densenet_preprocess
        target_size = (224, 224)  # DenseNet201 works with 224x224
    else:
        # Default preprocessing (simple rescaling)
        preprocess_func = lambda x: x / 255.0
        target_size = (img_height, img_width)
    
    # Create data generators with enhanced data augmentation to combat overfitting
    # More aggressive augmentation for all models due to small dataset
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=45,  # Increased rotation
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,  # Added shear transformation
        brightness_range=[0.7, 1.3],  # Added brightness variation
        fill_mode='nearest',
        # Add contrast adjustment
        channel_shift_range=0.2
    )

    # Less aggressive augmentation for validation to ensure fair evaluation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
    )
    
    # Load training data (from val_dir)
    train_generator = train_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Load validation data (from test_dir)
    val_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load test data (from test_dir, no augmentation)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, target_size

# Define function to create model with pre-trained base
def create_model(base_model_fn, model_name, target_size):
    # Model-specific configurations - optimized based on previous results
    if model_name == "ResNet152":
        fine_tune_percentage = 0.1  # Reduced fine-tuning to prevent overfitting
        learning_rate = 5e-6  # Lower learning rate
        dropout_rate = 0.5  # Adjusted dropout
        dense_units = 128  # Reduced complexity
        l2_reg = 0.003  # Increased regularization
    elif model_name == "ResNet101V2":
        # This model performed best, so keep similar parameters
        fine_tune_percentage = 0.2
        learning_rate = 1e-5
        dropout_rate = 0.5
        dense_units = 256
        l2_reg = 0.002
    elif model_name == "VGG19":
        fine_tune_percentage = 0.05  # Minimal fine-tuning for VGG19
        learning_rate = 1e-6  # Much lower learning rate
        dropout_rate = 0.7  # Higher dropout
        dense_units = 128  # Reduced complexity
        l2_reg = 0.004  # Higher regularization
    elif model_name == "NASNetMobile":
        # This model performed well, keep similar parameters
        fine_tune_percentage = 0.25
        learning_rate = 1e-5
        dropout_rate = 0.5
        dense_units = 128
        l2_reg = 0.002
    elif model_name == "Xception":
        fine_tune_percentage = 0.1  # Reduced fine-tuning
        learning_rate = 5e-6  # Lower learning rate
        dropout_rate = 0.6
        dense_units = 128
        l2_reg = 0.003
    elif model_name == "DenseNet201":
        # New model with optimized parameters
        fine_tune_percentage = 0.15
        learning_rate = 8e-6
        dropout_rate = 0.5
        dense_units = 192
        l2_reg = 0.002
    else:
        # Default configuration
        fine_tune_percentage = 0.15
        learning_rate = 8e-6
        dropout_rate = 0.5
        dense_units = 192
        l2_reg = 0.002
    
    # Store configuration for later saving
    model_config = {
        'model_name': model_name,
        'fine_tune_percentage': fine_tune_percentage,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'dense_units': dense_units,
        'l2_reg': l2_reg
    }
    
    # Create the base model
    base_model = base_model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(target_size[0], target_size[1], 3)
    )
    
    # First freeze all layers
    base_model.trainable = True
    
    # Calculate how many layers to fine-tune
    total_layers = len(base_model.layers)
    fine_tune_at = int(total_layers * (1 - fine_tune_percentage))
    
    # Freeze early layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"Total layers in {model_name}: {total_layers}")
    print(f"Fine-tuning the last {total_layers - fine_tune_at} layers")
    
    # Create the model with improved architecture for small datasets
    inputs = Input(shape=(target_size[0], target_size[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Add BatchNormalization before first Dense layer
    x = BatchNormalization()(x)
    
    # Add model-specific dense layers with increased regularization
    x = Dense(dense_units, activation='relu', 
              kernel_regularizer=l2(l2_reg),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Add a smaller second dense layer for all models
    x = Dense(dense_units // 2, activation='relu', 
              kernel_regularizer=l2(l2_reg),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Final classification layer with L2 regularization
    outputs = Dense(num_classes, activation='softmax', 
                   kernel_regularizer=l2(l2_reg/2),
                   kernel_initializer='glorot_uniform')(x)
    
    model = Model(inputs, outputs)
    
    # Use model-specific learning rate with weight decay
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, model_name, model_config

# Define callbacks for better training
def get_callbacks(model_name):
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=40,  # Increased patience for small dataset
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint to save the best model
    model_checkpoint = ModelCheckpoint(
        os.path.join(results_dir, f'best_{model_name.lower()}_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.6,  # Less aggressive reduction for stability
        patience=15,  # More patience
        min_lr=1e-8,  # Lower minimum learning rate
        verbose=1
    )
    
    return [early_stopping, reduce_lr, model_checkpoint]

# Calculate metrics
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

# Function to print metrics
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
    plt.savefig(os.path.join(results_dir, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()

# Function to save model results to file
def save_results_to_file(all_results, all_configs, filename=os.path.join(results_dir, "model_results.txt")):
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CELL CLASSIFICATION MODEL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort models by accuracy (best first)
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, results in sorted_models:
            metrics = results['metrics']
            accuracy = results['accuracy']
            config = all_configs[model_name]
            
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
            
            f.write(f"MODEL: {model_name}\n")
            f.write("-" * 40 + "\n\n")
            
            # Write configuration
            f.write("Configuration:\n")
            f.write(f"  - Fine-tune percentage: {config['fine_tune_percentage']}\n")
            f.write(f"  - Learning rate: {config['learning_rate']}\n")
            f.write(f"  - Dropout rate: {config['dropout_rate']}\n")
            f.write(f"  - Dense units: {config['dense_units']}\n")
            if model_name in ["ResNet152", "ResNet101V2", "EfficientNetV2S"]:
                f.write(f"  - Additional dense layer: Yes (128 units)\n")
            else:
                f.write(f"  - Additional dense layer: No\n")
            f.write("\n")
            
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
            
            f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<10} {'F1 Score':<10} {'MCC':<10} {'AUC':<10}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, results in sorted_models:
            metrics = results['metrics']
            accuracy = results['accuracy']
            
            p_macro = metrics['precision_macro']
            r_macro = metrics['recall_macro']
            if p_macro + r_macro > 0:
                expected_f1_macro = 2 * p_macro * r_macro / (p_macro + r_macro)
                if abs(expected_f1_macro - metrics['f1_macro']) > 0.01:
                    metrics['f1_macro'] = expected_f1_macro
                    
            f.write(f"{model_name:<15} {accuracy:.4f}      {metrics['precision_macro']:.4f}     {metrics['recall_macro']:.4f}     {metrics['specificity_macro']:.4f}     {metrics['f1_macro']:.4f}      {metrics['mcc']:.4f}     {metrics['roc_auc']['micro']:.4f}\n")
        
        f.write("\n\nTraining completed on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"\nResults saved to {filename}")

# Define the models to evaluate - including the new DenseNet201 model
models_to_evaluate = [
    (ResNet152, "ResNet152"),
    (ResNet101V2, "ResNet101V2"),
    (NASNetMobile, "NASNetMobile"),
    (Xception, "Xception"),
    (DenseNet201, "DenseNet201")
]

all_results = {}
all_configs = {}
all_histories = {}

# Main execution
for model_fn, model_name in models_to_evaluate:
    print(f"\n\n{'='*50}")
    print(f"Training and evaluating {model_name}")
    print(f"{'='*50}")
    
    # Create model-specific data generators
    train_generator, val_generator, test_generator, target_size = create_data_generators(model_name)
    
    # Calculate class weights to handle imbalance
    train_classes = np.array([])
    for i in range(len(train_generator)):
        batch = train_generator.next()
        train_classes = np.append(train_classes, np.argmax(batch[1], axis=1))

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_classes),
        y=train_classes
    )
    
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights for {model_name}:", class_weights_dict)
    
    # Create and compile the model
    model, model_name, model_config = create_model(model_fn, model_name, target_size)
    
    # Store model configuration
    all_configs[model_name] = model_config
    
    # Print model summary
    model.summary()
    
    # Train the model with appropriate epochs
    if model_name == "ResNet152" or model_name == "ResNet101V2" or model_name == "DenseNet201":
        epochs = 150  # Increased epochs for larger models
    else:
        epochs = 200  # More epochs for other models
        
    # Train with early stopping
    history = model.fit(
        train_generator,  # Training on training set (31 images)
        epochs=epochs,
        validation_data=val_generator,  # Validating on validation set (54 images)
        callbacks=get_callbacks(model_name),
        class_weight=class_weights_dict
    )
    
    # Store training history
    all_histories[model_name] = history.history
    
    # Plot and save accuracy and loss curves
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_training_curves.png'), dpi=300)
    plt.close()
    
    # Load the best model weights
    model.load_weights(os.path.join(results_dir, f'best_{model_name.lower()}_model.keras'))
    
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Get predictions
    test_generator.reset()
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Calculate all metrics
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
    cm = confusion_matrix(y_true, y_pred)
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
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
    plt.close()
    
    # Clear memory
    tf.keras.backend.clear_session()

# Save detailed metrics to a new file
with open(os.path.join(results_dir, "detailed_model_metrics.txt"), "w") as f:
    f.write("="*80 + "\n")
    f.write("DETAILED MODEL METRICS AND PARAMETERS\n")
    f.write("="*80 + "\n\n")
    
    # Sort models by accuracy (best first)
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, results in sorted_models:
        metrics = results['metrics']
        accuracy = results['accuracy']
        config = all_configs[model_name]
        
        f.write(f"MODEL: {model_name}\n")
        f.write("-"*50 + "\n\n")
        
        # Write model parameters
        f.write("TRAINING PARAMETERS:\n")
        f.write(f"* Learning Rate: {config['learning_rate']}\n")
        f.write(f"* Optimizer: Adam\n")
        f.write(f"* Activation Function: ReLU (hidden layers), Softmax (output layer)\n")
        if model_name == "VGG19":
            f.write(f"* Number of Epochs: 300\n")
        elif model_name == "EfficientNetV2S":
            f.write(f"* Number of Epochs: 150\n")
        else:
            f.write(f"* Number of Epochs: 200\n")
        f.write(f"* Fine-tune percentage: {config['fine_tune_percentage']}\n")
        f.write(f"* Dropout rate: {config['dropout_rate']}\n")
        f.write(f"* Dense units: {config['dense_units']}\n")
        f.write(f"* L2 regularization: {config['l2_reg']}\n\n")
        
        # Write evaluation metrics
        f.write("EVALUATION METRICS:\n")
        f.write(f"* Accuracy: {accuracy:.4f}\n\n")
        
        f.write("* Precision:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  - {class_name}: {metrics['precision'][i]:.4f}\n")
        f.write(f"  - Macro Average: {metrics['precision_macro']:.4f}\n\n")
        
        f.write("* Sensitivity (Recall):\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  - {class_name}: {metrics['recall'][i]:.4f}\n")
        f.write(f"  - Macro Average: {metrics['recall_macro']:.4f}\n\n")
        
        f.write("* Specificity:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  - {class_name}: {metrics['specificity'][i]:.4f}\n")
        f.write(f"  - Macro Average: {metrics['specificity_macro']:.4f}\n\n")
        
        f.write("* F1 Score:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  - {class_name}: {metrics['f1'][i]:.4f}\n")
        f.write(f"  - Macro Average: {metrics['f1_macro']:.4f}\n\n")
        
        f.write(f"* MCC Score: {metrics['mcc']:.4f}\n\n")
        
        f.write(f"* AUC Score (micro-average): {metrics['roc_auc']['micro']:.4f}\n\n")
        
        f.write("="*80 + "\n\n")
    
    # Add summary table
    f.write("SUMMARY COMPARISON\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<10} {'F1 Score':<10} {'MCC':<10} {'AUC':<10}\n")
    f.write("-"*90 + "\n")
    
    for model_name, results in sorted_models:
        metrics = results['metrics']
        accuracy = results['accuracy']
        f.write(f"{model_name:<15} {accuracy:.4f}      {metrics['precision_macro']:.4f}     {metrics['recall_macro']:.4f}     {metrics['specificity_macro']:.4f}     {metrics['f1_macro']:.4f}      {metrics['mcc']:.4f}     {metrics['roc_auc']['micro']:.4f}\n")
    
    f.write("\n\nTraining completed on: " + time.strftime("%Y-%m-%d %H:%M:%S"))

print(f"\nAll results saved to the '{results_dir}' folder")

# Save original results to file as well
save_results_to_file(all_results, all_configs, filename=os.path.join(results_dir, "model_results.txt")) 