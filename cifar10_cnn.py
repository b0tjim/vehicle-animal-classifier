import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to 0-1 range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# Build CNN model
print("\nBuilding CNN model...")
model = keras.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
print("\nModel Architecture:")
model.summary()

# Save model architecture to file
with open('model_architecture.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Model architecture saved to 'model_architecture.txt'")

# Train model
print("\nTraining model...")
history = model.fit(x_train, y_train, 
                    epochs=50,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=1)

# Evaluate on test set
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
print("\nGenerating training history plots...")
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history plot saved to 'training_history.png'")
plt.show()

# Save results to file
with open('results.txt', 'w') as f:
    f.write("CIFAR-10 CNN Classification Results\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    f.write(f"Training completed over {len(history.history['accuracy'])} epochs\n")
    f.write(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%\n")
    f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%\n")

print("Results saved to 'results.txt'")
print("\nAll files generated successfully!")
print("Files to include in your report:")
print("  1. cifar10_cnn.py (this code file)")
print("  2. model_architecture.txt (model summary)")
print("  3. training_history.png (accuracy/loss plots)")
print("  4. results.txt (final metrics)")
