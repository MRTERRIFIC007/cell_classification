import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
test_dir = 'Data/oral cancer 0521-0618_tag300_Test'
val_dir = 'Data/oral cancer 0521-0618_tag300_Val'

# Define image parameters
img_height, img_width = 150, 150  # Reduced size for faster training
batch_size = 32
num_classes = 3

# Define class names
class_names = ['5nMTG', 'Control', 'NonCancerOral']

# Create data generators with data augmentation for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Create a simple CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
epochs = 20
history = model.fit(
    val_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')

# Get predictions
test_generator.reset()
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_prob):
    # Convert to one-hot encoding for ROC curve
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Calculate macro averages
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate specificity for each class
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(num_classes):
        true_neg = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_pos = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0)
    
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
        'f1': f1,
        'mcc': mcc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

# Calculate all metrics
metrics = calculate_metrics(y_true, y_pred, y_pred_prob)

# Print metrics
print("\n===== Classification Metrics =====")
print("\nPer-class metrics:")
for i, class_name in enumerate(class_names):
    print(f"\nClass: {class_name}")
    print(f"Precision: {metrics['precision'][i]:.4f}")
    print(f"Recall: {metrics['recall'][i]:.4f}")
    print(f"Specificity: {metrics['specificity'][i]:.4f}")
    print(f"F1 Score: {metrics['f1'][i]:.4f}")

print("\nOverall metrics:")
print(f"Macro-average Precision: {metrics['precision_macro']:.4f}")
print(f"Macro-average Recall: {metrics['recall_macro']:.4f}")
print(f"Macro-average F1 Score: {metrics['f1_macro']:.4f}")
print(f"MCC Score: {metrics['mcc']:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, color, class_name in zip(range(num_classes), ['blue', 'red', 'green'], class_names):
    plt.plot(
        metrics['fpr'][i],
        metrics['tpr'][i],
        color=color,
        lw=2,
        label=f'ROC curve of {class_name} (AUC = {metrics["roc_auc"][i]:.2f})'
    )

# Plot micro-average ROC curve
plt.plot(
    metrics['fpr']["micro"],
    metrics['tpr']["micro"],
    color='deeppink',
    linestyle=':',
    linewidth=4,
    label=f'Micro-average ROC curve (AUC = {metrics["roc_auc"]["micro"]:.2f})'
)

# Plot diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
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
plt.savefig('confusion_matrix.png')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()